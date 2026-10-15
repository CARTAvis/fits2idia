/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

// TODO do we need these?
FastConverter::FastConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) : Converter(inputFileName, outputFileName, progress, zMips) {}

MemoryUsage FastConverter::calculateMemoryUsage() {
    MemoryUsage m;
    
    m.sizes["Main dataset"] = depth * height * width * sizeof(float);
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {depth, height, width}, zMips);
    m.sizes["XY stats"] = Stats::size({depth}, numBins);
    
    if (depth > 1) {
        m.sizes["Rotation"] = m.sizes["Main dataset"];
        m.sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        m.sizes["Z stats"] = Stats::size({height, width});
    }
    
    for (auto& kv : m.sizes) {
        m.total += kv.second;
    }
    
    if (depth > 1) {
        m.total -= std::min(m.sizes["Mipmaps"], m.sizes["Rotation"]);
        m.note = " (Rotated dataset and mipmaps are not allocated at the same time.)";
    }
    
    return m;
}

void FastConverter::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    const hsize_t pixelProgressStride = std::max((hsize_t)1, (hsize_t)(width * height / 100));
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    
    TIMER(timer.start("Allocate"););
    
    // Process one stokes at a time
    hsize_t cubeSize = depth * height * width;
    standardCube = new float[cubeSize];
    
    statsXY.createBuffers({depth});
    
    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
        statsZ.createBuffers({height, width});
    }
    
    mipMaps.createBuffers({depth, height, width});
    
    std::string timerLabelXYRotation = depth > 1 ? "XY statistics and rotation" : "XY statistics";

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        DEBUG(std::cout << "Processing Stokes " << currentStokes << "..." << std::endl;);
        PROGRESS("Stokes " << currentStokes << ":" << std::endl);

        // Read data into memory space
        TIMER(timer.start("Read"););
        DEBUG(std::cout << "+ Reading main dataset..." << std::flush;);
        // measuring I/O :
        auto start_io = std::chrono::high_resolution_clock::now();
        readFitsData(inputFilePtr, 0, currentStokes, cubeSize, standardCube, swapStokesFreqAxis);
        auto end_io = std::chrono::high_resolution_clock::now();
        auto duration_io = ms_d(end_io - start_io);
        total_io_ms += double(duration_io.count());
        std::cout << "I/O (readFitsData) for Stokes : " << currentStokes << " took " << duration_io.count() << " milliseconds." << std::endl;

        
        // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
        if (depth > 1) {
            TIMER(timer.start("Allocate"););
            rotatedCube = new float[cubeSize];
        }
        
        DEBUG(std::cout << " " << timerLabelXYRotation <<  "..." << std::flush;);
        PROGRESS("\tMain loop\t");
        TIMER(timer.start(timerLabelXYRotation););

        // First loop calculates stats for each XY slice and rotates the dataset        
        auto start1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");
            StatsCounter counterXY;
            
            auto& indexXY = i;
            std::function<void(float)> accumulate;
            
            auto lazy_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazy(val);
            };
            
            auto first_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazyFirst(val);
                accumulate = lazy_accumulate;
            };
            
            accumulate = first_accumulate;
            
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    auto sourceIndex = k + width * j + (height * width) * i;
                    auto destIndex = i + depth * j + (height * depth) * k;
                    auto& val = standardCube[sourceIndex];
                    
                    if (depth > 1) {
                        rotatedCube[destIndex] = val;
                    }
                    
                    // Accumulate XY stats
                    if (std::isfinite(val)) {
                        accumulate(val);
                    } else {
                        counterXY.accumulateNonFinite();
                    }
                }
            }
            
            // Final correction of XY min and max
            statsXY.copyStatsFromCounter(indexXY, height * width, counterXY);
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = ms_d(end1 - start1);
        double duration1_ms = double(duration1.count());
        std::cout << "BENCHMARKING : total pure-processing time of 1st pass (rotation and statsXY): " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += duration1_ms;

        
        PROGRESS(std::endl);

        if (depth > 1) {
            // Consolidate XY stats into XYZ stats
            DEBUG(std::cout << " XYZ statistics..." << std::flush;);
            PROGRESS("\tXYZ stats" << std::endl);
            TIMER(timer.start("XYZ statistics"););
            
            auto start1 = std::chrono::high_resolution_clock::now();
            StatsCounter counterXYZ;

            for (hsize_t i = 0; i < depth; i++) {
                auto& indexXY = i;
                statsXY.accumulateStatsToCounter(counterXYZ, indexXY);
            }

            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);

            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
            
            DEBUG(std::cout << " Z statistics... " << std::flush;);
            PROGRESS("\tZ stats\t\t");
            TIMER(timer.start("Z statistics"););

#pragma omp parallel for
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    StatsCounter counterZ;
                    
                    auto indexZ = k + j * width;
                    PROGRESS_DECIMATED(indexZ, pixelProgressStride, ".");
                    
                    for (hsize_t i = 0; i < depth; i++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto& val = standardCube[sourceIndex];

                        if (std::isfinite(val)) {
                            // Not lazy; too much risk of encountering an ascending / descending sequence.
                            counterZ.accumulateFinite(val);
                        } else {
                            counterZ.accumulateNonFinite();
                        }
                    }
                    
                    statsZ.copyStatsFromCounter(indexZ, depth, counterZ);
                }
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = ms_d(end1 - start1);
            double duration1_ms = double(duration1.count());
            std::cout << "BENCHMARKING : total pure-processing time of 2nd pass (statsZ for all (X,Y) pixels): " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
            total_pureprocessing_ms += duration1_ms;


            PROGRESS(std::endl);
        }

        // Third loop handles histograms
        
        DEBUG(std::cout << " Histograms..." << std::flush;);
        PROGRESS("\tHistograms\t");
        TIMER(timer.start("Histograms"););
        
        double cubeMin;
        double cubeMax;
        double cubeRange;
        bool cubeHist(false);
                    
        if (depth > 1) {
            cubeMin = statsXYZ.minVals[0];
            cubeMax = statsXYZ.maxVals[0];
            cubeRange = cubeMax - cubeMin;
            cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
        }

        start1 = std::chrono::high_resolution_clock::now();
        
        statsXY.clearHistogramBuffers();
        statsXYZ.clearHistogramBuffers();

#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");
            
            auto& indexXY = i;
            double chanMin = statsXY.minVals[indexXY];
            double chanMax = statsXY.maxVals[indexXY];
            double chanRange = chanMax - chanMin;
            
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
            
            if (!chanHist && !cubeHist) {
                continue; // skip the loop entirely
            }
            
            auto doChannelHistogram = [&] (float val) {
                // XY histogram
                statsXY.accumulateHistogram(val, chanMin, chanRange, i);
            };
            
            auto doCubeHistogram = [&] (float val) {
                // Partial XYZ histogram
                statsXYZ.accumulatePartialHistogram(val, cubeMin, cubeRange, i);
            };
            
            auto doNothing = [&] (float val) {
                UNUSED(val);
            };
            
            std::function<void(float)> channelHistogramFunc = doChannelHistogram;
            std::function<void(float)> cubeHistogramFunc = doCubeHistogram;
                        
            if (!chanHist) {
                channelHistogramFunc = doNothing;
            }
            
            if (!cubeHist) {
                cubeHistogramFunc = doNothing;
            }

            for (hsize_t j = 0; j < width * height; j++) {
                auto& val = standardCube[i * width * height + j];

                if (std::isfinite(val)) {
                    channelHistogramFunc(val);
                    cubeHistogramFunc(val);
                }
            } // end of XY loop
        } // end of parallel Z loop                
        
        if (depth > 1) {
            // Consolidate partial XYZ histograms into final histogram
            statsXYZ.consolidatePartialHistogram();
        }
        
        end1 = std::chrono::high_resolution_clock::now();
        duration1 = ms_d(end1 - start1);
        duration1_ms = double(duration1.count());
        std::cout << "BENCHMARKING : total pure-processing time of 3rd pass (channel and cube Histograms): " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += duration1_ms;
        
        PROGRESS(std::endl);

        DEBUG(std::cout << " Writing main and rotated datasets... " << std::flush;);
        PROGRESS("\tWrite data" << std::endl);
        TIMER(timer.start("Write"););
             
        start_io = std::chrono::high_resolution_clock::now();                      
        std::vector<hsize_t> memDims = {depth, height, width};
        std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
        std::vector<hsize_t> start = trimAxes({currentStokes, 0, 0, 0}, N);
        writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
        
        if (depth > 1) {
            // This all technically worked if we reused the standard filespace and memspace
            // But it's probably not a good idea to rely on two incorrect values cancelling each other out
            std::vector<hsize_t> swizzledCount = trimAxes({1, width, height, depth}, N);
            std::vector<hsize_t> swizzledMemDims = {width, height, depth};
            writeHdf5Data(swizzledDataSet, rotatedCube, swizzledMemDims, swizzledCount, start);
        }
        end_io = std::chrono::high_resolution_clock::now();
        duration_io = ms_d(end_io - start_io);
        total_io_ms += double(duration_io.count());


        // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
        if (depth > 1) {
            DEBUG(std::cout << " Freeing memory from rotated dataset..." << std::flush;);
            TIMER(timer.start("Free"););
            
            delete[] rotatedCube;
        }
        
        // Fourth loop handles mipmaps
        
        // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
        DEBUG(std::cout << " Mipmaps..." << std::endl;);
        PROGRESS("\tMipmaps\t\t");
        TIMER(timer.start("Mipmaps"););


        start1 = std::chrono::high_resolution_clock::now();                 
#pragma omp parallel for
        for (hsize_t c = 0; c < depth; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            for (hsize_t y = 0; y < height; y++) {
                for (hsize_t x = 0; x < width; x++) {
                    auto sourceIndex = x + width * y + (height * width) * c;
                    auto& val = standardCube[sourceIndex];
                    if (std::isfinite(val)) {
                        mipMaps.accumulate(val, x, y, c);
                    }
                }
            }
        } // end of mipmap loop
        PROGRESS(std::endl);
        
        // Final mipmap calculation
        mipMaps.calculate();
        
        end1 = std::chrono::high_resolution_clock::now();
        duration1 = ms_d(end1 - start1);
        duration1_ms = double(duration1.count());
        std::cout << "BENCHMARKING : total pure-processing time of 4th pass (MipMaps accum and calc): " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += duration1_ms;
        
        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats & mipmaps" << std::endl);
        
        start_io = std::chrono::high_resolution_clock::now();
        // Write the mipmaps
        mipMaps.write(currentStokes, 0);
        
        // Write the statistics                
        statsXY.write({1, depth}, {currentStokes, 0});
        
        if (depth > 1) {
            statsXYZ.write({1}, {currentStokes});
            statsZ.write({1, height, width}, {currentStokes, 0, 0});
        }
        end_io = std::chrono::high_resolution_clock::now();
        duration_io = ms_d(end_io - start_io);
        total_io_ms += double(duration_io.count());
                
        // Clear the mipmaps before the next Stokes
        TIMER(timer.start("Mipmaps"););
        mipMaps.resetBuffers();
        
    } // end of Stokes loop
    
    // total I/O time:
    std::cout << "BENCHMARKING: total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = ms_d(end - start);
    double duration_ms = double(duration.count());
    std::cout << "Execution of entire FastConverter::copyAndCalculate took " << duration_ms << " milliseconds " << duration_ms/1000.00 << " seconds" << std::endl;
    double unaccounted_for_ms = duration.count() - total_pureprocessing_ms - total_io_ms;
    std::cout << "Unaccounted for: " << unaccounted_for_ms/1000.00 << " seconds" << std::endl;

}

// I/O cost model for FastConverter -- mirrors copyAndCalculate() call by call. Everything is done
// one whole Stokes at a time, so every phase is one call (or one call per dataset/level) per Stokes:
//
//   FITS read of the whole Stokes cube -> standardDataSet write -> swizzledDataSet write
//   -> mipmap writes (all XY levels) -> statsXY write -> statsXYZ write -> statsZ write
//
// Same modelling conventions as SmartFastTwoPassConverter::estimateIO (layouts as created in
// Converter::convert(), on-disk element sizes), so the two estimates are directly comparable.
IOCostBreakdown FastConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                          hsize_t numBins,
                                          const IOCostModel& readModel,
                                          const IOCostModel& writeModel) {
    IOCostBreakdown result;

    // ---- on-disk layouts (as created in Converter::convert()) ----
    const std::vector<hsize_t> standardDims4 = {stokes, depth, height, width};
    const std::vector<hsize_t> swizzledDims4 = {stokes, width, height, depth};
    const std::vector<hsize_t> statsZDims    = {stokes, height, width};

    // standardDataSet: chunked {1,1,TILE,TILE} only if both image dims >= TILE_SIZE (useChunks)
    const std::vector<hsize_t> standardChunks =
        useChunks({height, width}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE} : std::vector<hsize_t>{};

    // swizzledDataSet: always contiguous for FAST -- Converter::convert() only chunks it (-C) for
    // SMART* and SLOW converters.
    const std::vector<hsize_t> swizzledChunks = {};

    // statsZ: chunked {1, min(TILE,H), min(TILE,W)}
    const std::vector<hsize_t> statsZChunks = {1, std::min((hsize_t)TILE_SIZE, height), std::min((hsize_t)TILE_SIZE, width)};

    // On-disk element sizes of the 5 basic-stats datasets: MIN, MAX, SUM, SUM_SQ are float,
    // NAN_COUNT is int64.
    const hsize_t basicElemSizes[] = {4, 4, 4, 4, 8};

    const hsize_t cubeBytes = depth * height * width * sizeof(float);

    {
        // readFitsData(..., cubeSize, ...): one contiguous read of the whole Stokes cube
        PhaseAccumulator acc;
        acc.add(repeatEstimate(IOOpEstimate{1, cubeBytes, cubeBytes}, stokes), readModel);
        result.phases.push_back(acc.toPhase("FITS read (whole Stokes cube)"));
    }
    {
        // writeHdf5Data(standardDataSet, ...) of {1, depth, height, width}
        PhaseAccumulator acc;
        auto e = estimateHyperslabIO(standardDims4, standardChunks, {1, depth, height, width}, sizeof(float));
        acc.add(repeatEstimate(e, stokes), writeModel);
        result.phases.push_back(acc.toPhase("standardDataSet write"));
    }
    if (depth > 1) {
        // writeHdf5Data(swizzledDataSet, ...) of {1, width, height, depth}: covers the whole Stokes
        // plane of a contiguous dataset -> a single contiguous run
        PhaseAccumulator acc;
        auto e = estimateHyperslabIO(swizzledDims4, swizzledChunks, {1, width, height, depth}, sizeof(float));
        acc.add(repeatEstimate(e, stokes), writeModel);
        result.phases.push_back(acc.toPhase("swizzledDataSet write (contiguous)"));
    }
    {
        // mipMaps.write(currentStokes, 0): one full-depth write per XY level. Levels mirror the MipMaps
        // constructor without Z-mips. Datasets are float on disk, chunked with tileDims only if the
        // level is >= TILE_SIZE in both dims (useChunks).
        // NOTE: with -z the Z-mip levels would add further writes; they are not modelled here.
        PhaseAccumulator acc;
        hsize_t hLevel = height, wLevel = width;
        int mipXY = 1;
        do {
            if (mipXY > 1) {
                std::vector<hsize_t> levelDims = {stokes, depth, hLevel, wLevel};
                std::vector<hsize_t> levelChunks =
                    useChunks({hLevel, wLevel}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE} : std::vector<hsize_t>{};
                auto e = estimateHyperslabIO(levelDims, levelChunks, {1, depth, hLevel, wLevel}, sizeof(float));
                acc.add(repeatEstimate(e, stokes), writeModel);
            }
            mipXY *= 2;
            hLevel = (hsize_t)std::ceil((float)hLevel / 2);
            wLevel = (hsize_t)std::ceil((float)wLevel / 2);
        } while (2 * wLevel > MIN_MIPMAP_SIZE || 2 * hLevel > MIN_MIPMAP_SIZE);
        result.phases.push_back(acc.toPhase("mipmap writes (all XY levels)"));
    }
    {
        // statsXY.write({1, depth}, {s, 0}): 5 basic datasets + channel histograms
        PhaseAccumulator acc;
        for (auto es : basicElemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, sizeof(int64_t));
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("statsXY write (basic + channel histograms)"));
    }
    if (depth > 1) {
        {
            // statsXYZ.write({1}, {s}): 5 single values + the cube histogram
            PhaseAccumulator acc;
            for (auto es : basicElemSizes) {
                auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
                acc.add(repeatEstimate(e, stokes), writeModel);
            }
            if (numBins > 0) {
                auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, sizeof(int64_t));
                acc.add(repeatEstimate(h, stokes), writeModel);
            }
            result.phases.push_back(acc.toPhase("statsXYZ write (basic + cube histogram)"));
        }
        {
            // statsZ.write({1, height, width}, {s, 0, 0}): whole image into each of the 5 chunked datasets
            PhaseAccumulator acc;
            for (auto es : basicElemSizes) {
                auto z = estimateHyperslabIO(statsZDims, statsZChunks, {1, height, width}, es);
                acc.add(repeatEstimate(z, stokes), writeModel);
            }
            result.phases.push_back(acc.toPhase("statsZ write"));
        }
    }

    return result;
}
