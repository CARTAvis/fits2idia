/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

SmartConverter::SmartConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) : Converter(inputFileName, outputFileName, progress, zMips) {}

MemoryUsage SmartConverter::calculateMemoryUsage() {
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

void SmartConverter::copyAndCalculate() {
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    hsize_t numTiles = std::ceil(width / TILE_SIZE) * std::ceil(height / TILE_SIZE);
    const hsize_t tileProgressStride = std::max((hsize_t)1, (hsize_t)(numTiles / 100));
    
    const hsize_t REGION_MULTIPLIER = 32;
    
    // Allocate one channel at a time, and no swizzled data
    hsize_t cubeSize = height * width;
    TIMER(timer.start("Allocate"););
    standardCube = new float[cubeSize];
    
    // Allocate one stokes of stats at a time
    statsXY.createBuffers({depth}, height);
    
    if (depth > 1) {
        statsXYZ.createBuffers({});
    }
    
    mipMaps.createBuffers({1, height, width});

    std::vector<hsize_t> count = trimAxes({1, 1, height, width}, N);
    std::vector<hsize_t> memDims = {height, width};
    
    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";

    
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        StatsCounter counterXYZ;
        
        for (hsize_t c = 0; c < depth; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            // read one channel
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube);
            
            // Write the standard dataset
            
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            
            std::vector<hsize_t> start = trimAxes({s, c, 0, 0}, N);
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            
            DEBUG(std::cout << " Accumulating XY stats and mipmaps..." << std::flush;);
            TIMER(timer.start(timerLabelStatsMipmaps););

            StatsCounter counterXY;
            
            auto indexXY = c;
            std::function<void(float)> accumulate;
            
            //
            
            auto lazy_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazy(val);
            };
            
            auto first_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazyFirst(val);
                accumulate = lazy_accumulate;
            };
            
            accumulate = first_accumulate;
            
            int mipIndex;
            StatsCounter counterRegion;
            
            int regionRows = std::ceil(height / (float)REGION_MULTIPLIER);
            int regionCols = std::ceil(width / (float)REGION_MULTIPLIER);
            int adjustedStandardCubeSize = regionRows * regionCols;
            
#pragma omp parallel for default(none) private (mipIndex, counterRegion) shared (standardCube, adjustedStandardCubeSize, mipMaps, counterXY, cubeSize, REGION_MULTIPLIER)
            for (mipIndex = 0; mipIndex < adjustedStandardCubeSize; mipIndex += 1 ) {
                counterRegion.reset();
                hsize_t x0,y0,z0;
                MipIndexToXYZ(mipIndex, x0, y0, z0, width, height, REGION_MULTIPLIER, REGION_MULTIPLIER, 1); //use index of higher-order mipmap-space to keep lower-order mipmaps thread-safe
                for (int y = y0; y < y0 + REGION_MULTIPLIER; y++) {
                    for (int x = x0; x < x0 + REGION_MULTIPLIER; x++) {
                        auto pos = y * width + x;
                        if (x >= width || y >= height) {    //check if we are out of bounds
                            continue;
                        }
                        //std::cout << "pos: " << pos << std::endl;
                        auto& val = standardCube[pos];
                        if (std::isfinite(val)) {
                            // region statistics
                            counterRegion.accumulateFinite(val);
                    
                            // Accumulate mipmaps
                            mipMaps.accumulate(val, x, y, 0); //This will not conflict with the other threads as regions are separate
                            
                        } else {
                            counterRegion.accumulateNonFinite();
                        }
                    }
                }
#pragma omp critical
                counterXY.accumulateFromCounter(counterRegion);      // Accumulate to slice's XY stats from thread-local X stats
            } // end of region loop
            
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
            
            
            // Write the mipmaps
            DEBUG(std::cout << " Writing mipmaps..." << std::flush;);
            TIMER(timer.start("Write"););
            mipMaps.write(s, c);
            
            // Reset mipmaps before next channel
            DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
            TIMER(timer.start(timerLabelStatsMipmaps););
            mipMaps.resetBuffers();
            
        } // end of first channel loop
        
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
            
            auto doChannelHistogram = [&] (float val, hsize_t offset) {
                // XY histogram
                statsXY.accumulatePartialHistogram(val, chanMin, chanRange, offset);
            };
            
            auto doCubeHistogram = [&] (float val) {
                // XYZ histogram
                statsXYZ.accumulateHistogram(val, cubeMin, cubeRange, 0);
            };
            
            auto doNothing = [&] (float val) {
                UNUSED(val);
            };
            
            auto doNothingOffset = [&] (float val, hsize_t offset) {
                UNUSED(val);
            };
            
            std::function<void(float,hsize_t)> channelHistogramFunc = doChannelHistogram;
            std::function<void(float)> cubeHistogramFunc = doCubeHistogram;
            
            if (!chanHist) {
                channelHistogramFunc = doNothingOffset;
            }
            
            if (!cubeHist) {
                cubeHistogramFunc = doNothing;
            }
            
            // read one channel
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube);

            DEBUG(std::cout << " Calculating histogram(s)..." << std::endl;);
            TIMER(timer.start("Histograms"););

            hsize_t y ;
#pragma omp parallel for default(none) private(y) shared (standardCube, height, width, channelHistogramFunc, cubeHistogramFunc)
            for (y = 0; y < height; y++) {
                for (hsize_t x = 0; x < width; x++) {
                    auto pos = y * width + x;
                    auto& val = standardCube[pos];
                    if (std::isfinite(val)) {
                        channelHistogramFunc(val, y);
                        cubeHistogramFunc(val);
                    }
                }
            } // end of XY loop
            
            statsXY.consolidatePartialHistogram(c);
            
        } // end of second channel loop (XY and XYZ histograms)
        
        PROGRESS(std::endl);
        
        // Write the statistics
        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats & mipmaps" << std::endl);
                
        statsXY.write({1, depth}, {s, 0});
        
        if (depth > 1) {
            statsXYZ.write({1}, {s});
        }
    
    } // end of stokes
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
            
    // Swizzle
    if (depth > 1) {
        DEBUG(std::cout << "Performing tiled rotation." << std::endl;);
        PROGRESS("Tiled rotation & Z stats" << std::endl);
        TIMER(timer.start("Allocate"););
        
        hsize_t sliceSize = product(trimAxes({stokes, depth, REGION_MULTIPLIER, REGION_MULTIPLIER}, N));
        float* standardSlice = new float[sliceSize];
        
        int numThreads = omp_get_max_threads();
        
        // Create a vector of float pointers for each thread
        std::vector<float*> rotatedSlices(numThreads);
        for (int i = 0; i < numThreads; i++) {
            rotatedSlices[i] = new float[sliceSize];
        }
        
        statsZ.createBuffers({height, width});
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            PROGRESS("\tStokes " << s << "\t");
            
            hsize_t tileCount(0);
            
            int mipIndex;
            
            int mipRows = std::ceil(height / (float)REGION_MULTIPLIER);
            int mipCols = std::ceil(width / (float)REGION_MULTIPLIER);
            int adjustedStandardCubeSize = mipRows * mipCols;
            
#pragma omp parallel for default(none) private (mipIndex) shared (s, statsZ, tileProgressStride, standardSlice, rotatedSlices, adjustedStandardCubeSize, mipMaps, tileCount, cubeSize, REGION_MULTIPLIER, std::cout)
            for (mipIndex = 0; mipIndex < adjustedStandardCubeSize; mipIndex += 1 ) {
                // Get the thread id
                int threadId = omp_get_thread_num();
                
                // Use the thread's own rotatedSlice
                float* rotatedSlice = rotatedSlices[threadId];
                
                hsize_t x0,y0,z0;
                MipIndexToXYZ(mipIndex, x0, y0, z0, width, height, REGION_MULTIPLIER, REGION_MULTIPLIER, 1); //use index of higher-order mipmap-space to keep lower-order mipmaps thread-safe
                hsize_t xOffset = x0;
                hsize_t yOffset = y0;
                tileCount++;
                hsize_t xSize = std::min(REGION_MULTIPLIER, width - xOffset);
                hsize_t ySize = std::min(REGION_MULTIPLIER, height - yOffset);
                
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
                
                //depth is going all the way, but ysize and xsize are not!
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
                
                // A separate pass over the same slice depth-last
                DEBUG(std::cout << " Calculating Z statistics..." << std::flush;);
                TIMER(timer.start("Z statistics"););
                StatsCounter counterZ;
                for (hsize_t j = 0; j < ySize; j++) {
                    for (hsize_t k = 0; k < xSize; k++) {
                        counterZ.reset();
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
                        hsize_t index = (xOffset + k) + width * (yOffset + j);
#pragma omp critical
                            statsZ.copyStatsFromCounter(index, depth, counterZ);
                    }
                }
                
                // write tile slice
                DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                TIMER(timer.start("Write"););
                
                auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);
                
                writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                
                DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
              
                
                }
                // write Z statistics
                statsZ.write();
            PROGRESS(std::endl);
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        for (int i = 0; i < numThreads; i++) {
            delete[] rotatedSlices[i];
        }
    }
}
