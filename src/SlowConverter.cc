/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

SlowConverter::SlowConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) : Converter(inputFileName, outputFileName, progress, zMips) {}

MemoryUsage SlowConverter::calculateMemoryUsage() {
    MemoryUsage m;

    m.sizes["Main dataset"] = height * width * sizeof(float);
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {1, height, width}, zMips);
    m.sizes["XY stats"] = Stats::size({depth}, numBins);
    
    if (depth > 1) {
        m.sizes["Rotation"] = 2 * product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N)) * sizeof(float);
        m.sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        m.sizes["Z stats"] = Stats::size({TILE_SIZE, TILE_SIZE});
    }
    
    for (auto& kv : m.sizes) {
        m.total += kv.second;
    }
    
    if (depth > 1) {
        m.total -= std::min(m.sizes["Main dataset"], m.sizes["Rotation"] + m.sizes["Z stats"]);
        m.note = " (Main dataset and slices for rotation and Z statistics are not allocated at the same time.)";
    }

    return m;
}

void SlowConverter::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    hsize_t numTiles = std::ceil(width / TILE_SIZE) * std::ceil(height / TILE_SIZE);
    const hsize_t tileProgressStride = std::max((hsize_t)1, (hsize_t)(numTiles / 100));
    
    // predict execution time:
    IOCostModel readModel, writeModel;
    get_io_cost_model("SETONIX", readModel, writeModel );
    IOCostBreakdown iocost = estimateIO(stokes, depth, height, width, numBins, readModel, writeModel );
    iocost.print();
    
    // Allocate one channel at a time, and no swizzled data
    hsize_t cubeSize = height * width;
    TIMER(timer.start("Allocate"););
    standardCube = new float[cubeSize];
    
    // Allocate one stokes of stats at a time
    statsXY.createBuffers({depth});
    
    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
    }
    
    mipMaps.createBuffers({1, height, width});

    std::vector<hsize_t> count = trimAxes({1, 1, height, width}, N);
    std::vector<hsize_t> memDims = {height, width};
    
    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        StatsCounter counterXYZ;

        double total_first_pass_processing_ms = 0.00;
        for (hsize_t c = 0; c < depth; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            // read one channel
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            
            // measuring I/O :
            auto start_io = std::chrono::high_resolution_clock::now();
            TIMER(timer.start("Read"););
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube, swapStokesFreqAxis);
            
            // Write the standard dataset
            
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););            
            
            std::vector<hsize_t> start = trimAxes({s, c, 0, 0}, N);
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for channel : " << c << " took " << duration_io.count() << " milliseconds." << std::endl;
            
            DEBUG(std::cout << " Accumulating XY stats and mipmaps..." << std::flush;);
            TIMER(timer.start(timerLabelStatsMipmaps););

            StatsCounter counterXY;
            auto indexXY = c;
            std::function<void(float)> accumulate;
            
            auto lazy_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazy(val);
            };
            
            auto first_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazyFirst(val);
                accumulate = lazy_accumulate;
            };
            
            accumulate = first_accumulate;

            auto start1 = std::chrono::high_resolution_clock::now();            
            for (hsize_t y = 0; y < height; y++) {
                for (hsize_t x = 0; x < width; x++) {
                    auto pos = y * width + x; // relative to channel slice
                    auto& val = standardCube[pos];
                                        
                    if (std::isfinite(val)) {
                        // XY statistics
                        accumulate(val);
                        
                        // Accumulate mipmaps
                        mipMaps.accumulate(val, x, y, 0);
                        
                    } else {
                        counterXY.accumulateNonFinite();
                    }
                }
            } // end of XY loop
            
            // Final correction of XY min and max
            DEBUG(std::cout << " Final XY stats..." << std::flush;);
            statsXY.copyStatsFromCounter(indexXY, height * width, counterXY);
            
            // Accumulate XYZ statistics
            if (depth > 1) {
                DEBUG(std::cout << " Accumulating XYZ stats..." << std::flush;);
                statsXY.accumulateStatsToCounter(counterXYZ, indexXY);
            }
            
            // Final mipmap calculation
            DEBUG(std::cout << " Final mipmaps..." << std::flush;);
            mipMaps.calculate();
            
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << "Execution of 1st loop took: " << duration1.count() << " milliseconds." << std::endl;
            total_first_pass_processing_ms += double(duration1.count());
            
            // Write the mipmaps
            DEBUG(std::cout << " Writing mipmaps..." << std::flush;);
            TIMER(timer.start("Write"););
            start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(s, c);
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            
            // Reset mipmaps before next channel
            DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
            TIMER(timer.start(timerLabelStatsMipmaps););
            mipMaps.resetBuffers();
            
        } // end of first channel loop
        std::cout << "BENCHMARKING : total pure-processing time of 1st pass: " << total_first_pass_processing_ms << " milliseconds " << float(total_first_pass_processing_ms)/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += total_first_pass_processing_ms;
        
        PROGRESS(std::endl);
        
        if (depth > 1) {
            // Final correction of XYZ min and max
            DEBUG(std::cout << " Final XYZ stats..." << std::flush;);
            PROGRESS("\tXYZ stats" << std::endl);
            TIMER(timer.start(timerLabelStatsMipmaps););
            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);
        }
        
        // XY and XYZ histograms
        // We need a second pass over all channels because we need cube min and max (and channel min and max per channel)
        // We do the second pass backwards to take advantage of caching
        DEBUG(std::cout << " Histograms..." << std::endl;);
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
        
        statsXY.clearHistogramBuffers();
        statsXYZ.clearHistogramBuffers();
        
        DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);
        
        // second pass:
        double total_second_pass_processing_ms = 0.00;
        auto start2 = std::chrono::high_resolution_clock::now();
        for (hsize_t c = depth; c-- > 0; ) {
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            auto indexXY = c;
                            
            double chanMin = statsXY.minVals[indexXY];
            double chanMax = statsXY.maxVals[indexXY];
            double chanRange = chanMax - chanMin;
            
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
            DEBUG(std::cout << " Will " << (chanHist ? "" : "not ") << "calculate channel histogram." << std::flush;);
            
            if (!chanHist && !cubeHist) {
                continue;
            }
            
            auto doChannelHistogram = [&] (float val) {
                // XY histogram
                statsXY.accumulateHistogram(val, chanMin, chanRange, c);
            };
            
            auto doCubeHistogram = [&] (float val) {
                // XYZ histogram
                statsXYZ.accumulateHistogram(val, cubeMin, cubeRange, 0);
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
            
            // read one channel
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            
            auto start_io = std::chrono::high_resolution_clock::now();
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube, swapStokesFreqAxis);
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            std::cout << "2nd I/O (readFitsData) for channel : " << c << " took " << duration_io.count() << " milliseconds." << std::endl;

            DEBUG(std::cout << " Calculating histogram(s)..." << std::endl;);
            TIMER(timer.start("Histograms"););
            
            auto start1 = std::chrono::high_resolution_clock::now();
            for (hsize_t p = 0; p < width * height; p++) {
                auto& val = standardCube[p];
                    if (std::isfinite(val)) {
                        channelHistogramFunc(val);
                        cubeHistogramFunc(val);
                    }
            } // end of XY loop
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            total_second_pass_processing_ms += double(duration1.count());
            std::cout << "Execution of XY-loop for channel" << c << " took: " << duration1.count() << " milliseconds." << std::endl;
        } // end of second channel loop (XY and XYZ histograms)
        total_pureprocessing_ms += total_second_pass_processing_ms;
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        std::cout << "Execution of 2nd big standard conversion loop over all channels took: " << duration2.count() << " milliseconds." << std::endl;
        std::cout << "BENCHMARKING : total pure-processing time of 2nd pass: " << total_second_pass_processing_ms << " milliseconds " << (float(total_second_pass_processing_ms)/1000.00) << " seconds" << std::endl;
        
        PROGRESS(std::endl);
        
        // Write the statistics
        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats & mipmaps" << std::endl);
                
        statsXY.write({1, depth}, {s, 0});
        
        if (depth > 1) {
            statsXYZ.write({1}, {s});
        }
    
    } // end of stokes
    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds." << std::endl;
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
            
    // Swizzle
    if (depth > 1) {
        DEBUG(std::cout << "Performing tiled rotation." << std::endl;);
        PROGRESS("Tiled rotation & Z stats" << std::endl);
        TIMER(timer.start("Allocate"););
        
        hsize_t sliceSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];
        
        statsZ.createBuffers({TILE_SIZE, TILE_SIZE});
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            PROGRESS("\tStokes " << s << "\t");
            
            hsize_t tileCount(0);

            double total_rotation_pass_processing_ms = 0.00;
            auto start1 = std::chrono::high_resolution_clock::now();            
            for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                    auto starttile = std::chrono::high_resolution_clock::now();
                    tileCount++;
                    hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
                    hsize_t ySize = std::min(TILE_SIZE, height - yOffset);
                    
                    DEBUG(std::cout << "+ Processing tile slice at " << xOffset << ", " << yOffset << "..." << std::flush;);
                    PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");
                    
                    // read tile slice
                    DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                    TIMER(timer.start("Read"););
                    
                    auto standardMemDims = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardCount = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardStart = trimAxes({s, 0, yOffset, xOffset}, N);
                    
                    readHdf5Data(standardDataSet, standardSlice, standardMemDims, standardCount, standardStart);
                    
                    // rotate tile slice
                    DEBUG(std::cout << " Calculating rotation..." << std::flush;);
                    TIMER(timer.start("Rotation"););
                                        
                    auto start2 = std::chrono::high_resolution_clock::now();
                    for (hsize_t i = 0; i < depth; i++) {
                        for (hsize_t j = 0; j < ySize; j++) {
                            for (hsize_t k = 0; k < xSize; k++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto& val = standardSlice[sourceIndex];
                                
                                // rotation
                                auto destIndex = i + depth * j + (ySize * depth) * k;
                                rotatedSlice[destIndex] = val;
                            }
                        }
                    }
                    auto end2 = std::chrono::high_resolution_clock::now();
                    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
                    std::cout << "Execution of small rotation-loop took: " << duration2.count() << " milliseconds." << std::endl;
                    
                    // A separate pass over the same slice depth-last 
                    DEBUG(std::cout << " Calculating Z statistics..." << std::flush;);
                    TIMER(timer.start("Z statistics"););
                    
                    auto start3 = std::chrono::high_resolution_clock::now();
                    for (hsize_t j = 0; j < ySize; j++) {
                        for (hsize_t k = 0; k < xSize; k++) {
                            StatsCounter counterZ;
                            auto indexZ = k + xSize * j;
                            
                            for (hsize_t i = 0; i < depth; i++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto& val = standardSlice[sourceIndex];
                                
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
                    auto end3 = std::chrono::high_resolution_clock::now();
                    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
                    std::cout << "Execution of counter/stats-Z loop took: " << duration3.count() << " milliseconds." << std::endl;
                    auto duration_processing = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start2);
                    total_rotation_pass_processing_ms += double(duration_processing.count());
                    
                    // write tile slice
                    DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                    TIMER(timer.start("Write"););
                    
                    auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);
                    
                    auto start_io = std::chrono::high_resolution_clock::now();
                    writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                    auto end_io = std::chrono::high_resolution_clock::now();
                    auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    DEBUG(std::cout << "3rd I/O (writeHdf5Data) for xOffset = " << xOffset << " yOffset = " << yOffset  << " took " << duration_io.count() << " milliseconds." << std::endl;);
                    
                    start_io = std::chrono::high_resolution_clock::now();
                    DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
                    // write Z statistics
                    statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
                    end_io = std::chrono::high_resolution_clock::now();
                    duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    DEBUG(std::cout << "4th I/O (statsZ.write) for xOffset = " << xOffset << " yOffset = " << yOffset  << " took " << duration_io.count() << " milliseconds." << std::endl;);
                    std::cout << "Is this printed ???" << std::endl;
                    
                    auto endtile = std::chrono::high_resolution_clock::now();
                    auto durationtile = std::chrono::duration_cast<std::chrono::milliseconds>(endtile - starttile);
                    std::cout << "Execution of rotation of 1 tile, including writting, took: " << durationtile.count() << " milliseconds." << std::endl;
                }
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << "Execution of loop over X/Y Offsets for Stokes = " << s << " took: " << duration1.count() << " milliseconds." << std::endl;
            std::cout << "BENCHMARKING : total pure-processing time of rotation pass: " << total_rotation_pass_processing_ms << " milliseconds "
                      << (float(total_rotation_pass_processing_ms)/1000.0) << " seconds" << std::endl;
            total_pureprocessing_ms += total_rotation_pass_processing_ms;

            PROGRESS(std::endl);
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }

    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds." << std::endl;
    std::cout << "BENCHMARKING : total pure-processing time of 1st, 2nd and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of entire SlowConverter::copyAndCalculate took: " << duration.count() << " milliseconds." << std::endl;
}

IOCostBreakdown SlowConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel)
{
    IOCostBreakdown result;

    std::vector<hsize_t> standardDims = {stokes, depth, height, width};
    std::vector<hsize_t> chunkDims    = {1, 1, TILE_SIZE, TILE_SIZE};

    // ---------- 1st pass: SlowConverter.cc lines ~71-164 ----------

    // readFitsData(...) once per channel: not a hyperslab call (cfitsio),
    // one contiguous image plane per call -- uniform size, single add() is fine.
    {
        PhaseAccumulator acc;
        hsize_t bytesPerOp = height * width * sizeof(float);
        acc.add({ stokes * depth, bytesPerOp, stokes * depth * bytesPerOp }, readModel);
        result.phases.push_back(acc.toPhase("1st pass: FITS channel read (readFitsData)"));
    }

    // writeHdf5Data(standardDataSet, ...) once per channel — chunked, so this
    // decomposes into one write per {TILE_SIZE x TILE_SIZE} chunk. Boundary
    // chunks (where height/width isn't an exact multiple of TILE_SIZE) are
    // smaller than interior chunks; estimateHyperslabIO averages this single
    // call's chunks together, so predictSeconds sees one averaged size here.
    // If height/width are far from a TILE_SIZE multiple and you need this
    // exact, loop per-tile the way the 3rd pass already does below.
    {
        PhaseAccumulator acc;
        auto perChannel = estimateHyperslabIO(standardDims, chunkDims,
                                               {1, 1, height, width}, sizeof(float));
        acc.add(repeatEstimate(perChannel, stokes * depth), writeModel);
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet write"));
    }

    // mipMaps.write(s, c) once per channel, per level. Each level is a
    // different size (double-precision buffer, shrinking dims) -- cost each
    // level separately, then sum.
    {
        PhaseAccumulator acc;
        hsize_t mipXY = 1, hLevel = height, wLevel = width;
        do {
            mipXY *= 2;
            hLevel = (hsize_t)std::ceil((double)height / mipXY);
            wLevel = (hsize_t)std::ceil((double)width  / mipXY);
            IOOpEstimate perLevel{ 1, hLevel * wLevel * sizeof(double), hLevel * wLevel * sizeof(double) };
            acc.add(repeatEstimate(perLevel, stokes * depth), writeModel);
        } while (2 * wLevel > MIN_MIPMAP_SIZE || 2 * hLevel > MIN_MIPMAP_SIZE);
        result.phases.push_back(acc.toPhase("1st pass: mipmap writes (all levels)"));
    }

    // ---------- 2nd pass: SlowConverter.cc lines ~205-289 ----------

    {
        PhaseAccumulator acc;
        hsize_t bytesPerOp = height * width * sizeof(float);
        acc.add({ stokes * depth, bytesPerOp, stokes * depth * bytesPerOp }, readModel);
        result.phases.push_back(acc.toPhase("2nd pass: FITS channel read (readFitsData)"));
    }

    // statsXY.write(...) once per stokes: MIN/MAX(float,4B), SUM/SUMSQ(double,8B),
    // NAN_COUNT(int64,8B), plus HISTOGRAM(int64,8B) if numBins > 0. Different
    // element sizes -> cost each sub-dataset separately.
    {
        PhaseAccumulator acc;
        hsize_t elemSizes[] = {4, 4, 8, 8, 8}; // MIN, MAX, SUM, SUMSQ, NAN_COUNT
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, 8);
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("2nd pass: statsXY write"));
    }

    if (depth > 1) {
        PhaseAccumulator acc;
        hsize_t elemSizes[] = {4, 4, 8, 8, 8};
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, 8);
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("2nd pass: statsXYZ write"));
    }

    // ---------- 3rd pass: tiled rotation, SlowConverter.cc lines ~300-436 ----------
    if (depth > 1) {
        std::vector<hsize_t> swizzledDims = {stokes, width, height, depth};
        std::vector<hsize_t> statsZDims   = {stokes, height, width};

        PhaseAccumulator readAcc, writeAcc, statsZAcc;

        // Mirrors the real tile grid loop exactly, so boundary tiles (smaller
        // than TILE_SIZE) get their own, correctly-sized cost lookup instead
        // of being averaged in with interior tiles.
        for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
            hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
            for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                hsize_t ySize = std::min(TILE_SIZE, height - yOffset);

                auto r = estimateHyperslabIO(standardDims, chunkDims,
                                              {1, depth, ySize, xSize}, sizeof(float));
                readAcc.add(repeatEstimate(r, stokes), readModel);

                auto w = estimateHyperslabIO(swizzledDims, {},
                                              {1, xSize, ySize, depth}, sizeof(float));
                writeAcc.add(repeatEstimate(w, stokes), writeModel);

                hsize_t elemSizes[] = {4, 4, 8, 8, 8};
                for (auto es : elemSizes) {
                    auto s = estimateHyperslabIO(statsZDims, {}, {1, ySize, xSize}, es);
                    statsZAcc.add(repeatEstimate(s, stokes), writeModel);
                }
            }
        }

        result.phases.push_back(readAcc.toPhase("3rd pass: standardDataSet tile read"));
        result.phases.push_back(writeAcc.toPhase("3rd pass: swizzledDataSet tile write"));
        result.phases.push_back(statsZAcc.toPhase("3rd pass: statsZ tile write"));
    }

    return result;

}