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
    auto start = std::chrono::high_resolution_clock::now();
    
    if(memoryLimitInMb <= 0) {
       std::cerr << "ERROR : memory limit not set for the SmartConverter and it is strictly required -> exiting SmartConverter::copyAndCalculate function" << std::endl;
       return;
    }
    
    // calculate memory limits in different units, slice here is image in a single freq. channel:
    double memoryLimitInBytes = double(memoryLimitInMb) * 1024.00 * 1024.00;
    double double_memoryLimitInBytes = double(memoryLimitInMb) * 1024.0 * 1024.0;
    double sliceSizeInPixels = height * width;
    double memoryLimitInPixels = memoryLimitInBytes / sizeof(float);
    std::cout << "DEBUG memoryLimitInMb = " << memoryLimitInMb << " -> memoryLimitInBytes = " << memoryLimitInBytes << " -> memoryLimitInPixels = " << memoryLimitInPixels << std::endl;
    std::cout << "DEBUG double_memoryLimitInBytes = " << double_memoryLimitInBytes << std::endl;
    double memoryLimitInSlices = std::ceil(memoryLimitInPixels / sliceSizeInPixels);
    // first use MAX of memoryLimitInSlices and 1 , and then make sure we are not trying to read more channels than exist -> min(depth, MAX)
    int sliceIncrement = std::max(memoryLimitInSlices, (double)1.00); // first make sure we use at least 1 slice
    sliceIncrement = std::min( int(depth), sliceIncrement );          // then make sure we do not read more channels than there are in FITS file 
    // playing it safe and only using 1/2 of memory :
    if (sliceIncrement > 2) {
       std::cout << "DEBUG : sliceIncrement = " << sliceIncrement << " but playing it safe and using only half of it -> sliceIncrement := " << sliceIncrement/2 << std::endl;
       sliceIncrement = sliceIncrement/2;
    }
    std::cout << "DEBUG : final sliceIncrement = " << sliceIncrement << std::endl;
    int sliceIncrementCount = depth/sliceIncrement;                   // number of portions to be read 
    // int leftOverSlices = (depth - sliceIncrementCount*sliceIncrement);
    int leftOverSlices = (depth % sliceIncrement);    
    std::cout << "SIZEOF(float) = " << sizeof(float) << ", Image size:" << height << " x " << width << std::endl;
    std::cout << "MEMORY limits " << memoryLimitInMb << " MB = " << memoryLimitInPixels << " pixels = " << memoryLimitInSlices 
              << " slices -> sliceIncrement = " << sliceIncrement << " sliceIncrementCount = " << depth << "/" << sliceIncrement << " = " << sliceIncrementCount 
              << " -> leftover slices = " << leftOverSlices 
              << std::endl;
    // 
    int n_blocks = sliceIncrementCount;
    if (leftOverSlices > 0) {
       n_blocks++;
    }
    std::cout << "DEBUG : n_blocks = " << n_blocks << " vs.  sliceIncrementCount = " << sliceIncrementCount << " and leftOverSlices = " << leftOverSlices << std::endl;
              
              
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    hsize_t numTiles = std::ceil(width / TILE_SIZE) * std::ceil(height / TILE_SIZE);
    const hsize_t tileProgressStride = std::max((hsize_t)1, (hsize_t)(numTiles / 100));
    
    // Allocate one channel at a time, and no swizzled data
    hsize_t cubeSize = height * width;
    TIMER(timer.start("Allocate"););
    standardCube = new float[height * width * sliceIncrement];
    
    // Allocate one stokes of stats at a time
    statsXY.createBuffers({depth});
    
    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
    }
    
    mipMaps.createBuffers({1, height, width});

    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";


    double total_io_ms = 0.00;    
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        StatsCounter counterXYZ;
        
        for (hsize_t block = 0; block < n_blocks; block++ ) {
            hsize_t c_start = block*sliceIncrement;
            hsize_t c_end   = c_start + sliceIncrement;
            if (block == (n_blocks-1) && leftOverSlices > 0 ) {
               c_end   = c_start + leftOverSlices; // if last block and there are left over slices                
            }
            hsize_t n_channels = (c_end-c_start);
            hsize_t block_size = n_channels*height*width;
            std::cout << "DEBUG : processing block = " << block << " -> channel range " << c_start << " - " << c_end << std::endl;
            
            std::cout << "DEBUG : reading from c_start = " << c_start << " blockSize = " <<  block_size << " bytes" << std::endl;
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            // measuring I/O :
            auto start_io = std::chrono::high_resolution_clock::now();
            readFitsData(inputFilePtr, c_start, s, block_size, standardCube, swapStokesFreqAxis);

            // Write the standard dataset
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            
            std::vector<hsize_t> count = trimAxes({1, n_channels, height, width}, N); // ok as in original SmartConverter.cc it is trimAxes({1, sliceIncrement, height, width}, N);
            std::vector<hsize_t> memDims = {n_channels, height, width};
            std::vector<hsize_t> start = trimAxes({s, c_start, 0, 0}, N); // std::vector<hsize_t> start = trimAxes({s, c, 0, 0}, N);                       
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;


            
        
        for(hsize_t c = c_start; c < c_end; c++) {                 
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            // read one channel
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
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
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << "Execution of 1st loop took: " << duration1.count() << " milliseconds." << std::endl;
            
            
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
        } // end of loop over blocks
        
        
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
            std::cout << "DEBUG : reading from c = " << c << " cubeSize = " << cubeSize << std::endl;
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
            std::cout << "Execution of XY-loop for channel" << c << " took: " << duration1.count() << " milliseconds." << std::endl;
        } // end of second channel loop (XY and XYZ histograms)
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        std::cout << "Execution of 2nd big standard conversion loop over all channels took: " << duration2.count() << " milliseconds." << std::endl;

        
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
        
        hsize_t sliceSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];
        
        statsZ.createBuffers({TILE_SIZE, TILE_SIZE});
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            PROGRESS("\tStokes " << s << "\t");
            
            hsize_t tileCount(0);

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
                    
                    auto start_io = std::chrono::high_resolution_clock::now();
                    readHdf5Data(standardDataSet, standardSlice, standardMemDims, standardCount, standardStart);
                    auto end_io = std::chrono::high_resolution_clock::now();
                    auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    std::cout << "3nd I/O (readHdf5Data) for xOffset/yOffset : " << xOffset << " , " << yOffset << " took " << duration_io.count() << " milliseconds." << std::endl;
                    
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
                    
                    // write tile slice
                    DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                    TIMER(timer.start("Write"););
                    
                    auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);
                    
                    start_io = std::chrono::high_resolution_clock::now();
                    writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                    end_io = std::chrono::high_resolution_clock::now();
                    duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    std::cout << "4th I/O (writeHdf5Data) for xOffset/yOffset : " << xOffset << " , " << yOffset << " took " << duration_io.count() << " milliseconds." << std::endl;
                    
                    DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
                    // write Z statistics
                    statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
                    
                    auto endtile = std::chrono::high_resolution_clock::now();
                    auto durationtile = std::chrono::duration_cast<std::chrono::milliseconds>(endtile - starttile);
                    std::cout << "Execution of rotation of 1 tile, including writting, took: " << durationtile.count() << " milliseconds." << std::endl;
                }
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << "Execution of loop over X/Y Offsets for Stokes = " << s << " took: " << duration1.count() << " milliseconds." << std::endl;

            PROGRESS(std::endl);
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds." << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of entire SmartConverter::copyAndCalculate took: " << duration.count() << " milliseconds." << std::endl;
}
