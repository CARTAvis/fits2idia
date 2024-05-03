/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

SmartConverter::SmartConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMip, int memoryLimitInMb = 0) : Converter(inputFileName, outputFileName, progress, zMips)
{
    this->memoryLimitInMb = memoryLimitInMb;
}

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
    
    // 32 has produced best results in testing
    const hsize_t REGION_MULTIPLIER = 32;
    
    // Allocate one batch of slices at a time, and no swizzled data.
    // Batch size in slices determined by memory limit.
  
    hsize_t memoryLimitInBytes = memoryLimitInMb * 1024 * 1024;
    hsize_t sliceSizeInPixels = height * width;
    hsize_t memoryLimitInPixels = memoryLimitInBytes / sizeof(float);
    hsize_t memoryLimitInSlices = std::ceil((float)memoryLimitInPixels / (float)sliceSizeInPixels);
    
    // Increment over the full depth if the memory size limit is larger than the depth
    hsize_t sliceIncrement = std::min(memoryLimitInSlices, depth);
    
    hsize_t cubeSize = height * width * sliceIncrement;
    TIMER(timer.start("Allocate"););
    standardCube = new float[cubeSize];
    
    // Allocate one stokes of stats at a time
    statsXY.createBuffers({depth}, height);
    
    if (depth > 1) {
        statsXYZ.createBuffers({}, height);
    }
    
    mipMaps.createBuffers({1, height, width});

    std::vector<hsize_t> count = trimAxes({1, sliceIncrement, height, width}, N);
    std::vector<hsize_t> memDims = {sliceIncrement, height, width};
    
    
    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";

    
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        StatsCounter counterXYZ;
        
        // "working" versions of these variables are used to handle the last increment of slices that may be smaller than the batch size
        hsize_t workingIncrement = sliceIncrement;
        hsize_t workingCubeSize = cubeSize;
        std::vector<hsize_t> workingCount = count;
        std::vector<hsize_t> workingMemDims = memDims;
        
        for (hsize_t c = 0; c < depth; c = c + workingIncrement) {
            hsize_t leftOverSlices = depth - c;
            size_t actualIncrement = std::min(sliceIncrement, leftOverSlices);
            
            // If the last increment is smaller than the batch size, adjust the "working" variables
            if (actualIncrement < sliceIncrement) {
                workingIncrement = leftOverSlices;
                workingCubeSize = height * width * leftOverSlices;
                workingCount = trimAxes({1, leftOverSlices, height, width}, N);
                workingMemDims = {leftOverSlices, height, width};
            }
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            // read one channel
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            readFitsData(inputFilePtr, c, s, workingCubeSize, standardCube);
            
            // Write the standard dataset
            
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            
            std::vector<hsize_t> start = trimAxes({s, c, 0, 0}, N);
            writeHdf5Data(standardDataSet, standardCube, workingMemDims, workingCount, start);
            
            DEBUG(std::cout << " Accumulating XY stats and mipmaps..." << std::flush;);
            TIMER(timer.start(timerLabelStatsMipmaps););

            for(int slice = 0; slice < actualIncrement; slice++) {
            
            StatsCounter counterXY;
            
            auto indexXY = c + slice;
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
            
            int regionIndex;
            StatsCounter counterRegion;
            
            int regionRows = std::ceil((float)height / (float)REGION_MULTIPLIER);
            int regionCols = std::ceil((float)width / (float)REGION_MULTIPLIER);
            int cubeSizeInRegions = regionRows * regionCols;
            
#pragma omp parallel for default(none) private (regionIndex, counterRegion) shared (slice, standardCube, cubeSizeInRegions, mipMaps, counterXY, cubeSize, REGION_MULTIPLIER)
            for (regionIndex = 0; regionIndex < cubeSizeInRegions; regionIndex += 1 ) {
                counterRegion.reset();
                hsize_t x0,y0,z0;
                RegionIndexToXYZ(regionIndex, x0, y0, z0, width, height, REGION_MULTIPLIER, REGION_MULTIPLIER,
                    1); //use index of higher-order mipmap-space to keep lower-order mipmaps thread-safe
                for (hsize_t y = y0; y < y0 + REGION_MULTIPLIER; y++) {
                    for (hsize_t x = x0; x < x0 + REGION_MULTIPLIER; x++) {
                        auto pos = y * width + x + slice * width * height;
                        if (x >= width || y >= height) {    //check if we are out of bounds
                            continue;
                        }
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
            mipMaps.write(s, c + slice);
            
            // Reset mipmaps before next channel
            DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
            TIMER(timer.start(timerLabelStatsMipmaps););
            mipMaps.resetBuffers();
            
        }
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
        
        hsize_t startingSlice = depth - workingIncrement;
        for (hssize_t c = startingSlice; c > -1; c = c - sliceIncrement) {
            
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            
            // skip first batch of slices to prevent unnecessary file read
            if (c < startingSlice) {
                DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                TIMER(timer.start("Read"););
                readFitsData(inputFilePtr, c, s, workingCubeSize, standardCube);
            }
            
            DEBUG(std::cout << " Calculating histogram(s)..." << std::endl;);
            TIMER(timer.start("Histograms"););


        for (hsize_t slice = 0; slice < workingIncrement; slice++) {
            
            auto indexXY = c + slice;
            
            workingCubeSize = height * width * workingIncrement;
            
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
            
            auto doCubeHistogram = [&] (float val, hsize_t offset) {
                // XYZ histogram
                statsXYZ.accumulatePartialHistogram(val, cubeMin, cubeRange, offset);
            };
            
            auto doNothing = [&] (float val) {
                UNUSED(val);
            };
            
            auto doNothingOffset = [&] (float val, hsize_t offset) {
                UNUSED(val);
            };
            
            std::function<void(float,hsize_t)> channelHistogramFunc = doChannelHistogram;
            std::function<void(float,hsize_t)> cubeHistogramFunc = doCubeHistogram;
            
            if (!chanHist) {
                channelHistogramFunc = doNothingOffset;
            }
            
            if (!cubeHist) {
                cubeHistogramFunc = doNothingOffset;
            }
            
            hsize_t y ;
            
#pragma omp parallel for default(none) private(y) shared(slice, standardCube, height, width, channelHistogramFunc, cubeHistogramFunc)
                for (y = 0; y < height; y++) {
                    for (hsize_t x = 0; x < width; x++) {
                        auto pos = y * width + x + slice * width * height;
                        auto& val = standardCube[pos];
                        if (std::isfinite(val)) {
                            channelHistogramFunc(val, y);
                            cubeHistogramFunc(val, y);
                        }
                    }
                } // end of XY loop

                statsXY.consolidateAndClearPartialHistogram(c + slice);
                statsXYZ.consolidateAndClearPartialHistogram(0);
            } //for loop of sliceincrement ends here
            workingIncrement = sliceIncrement; //reset workingIncrement to sliceIncrement for remainder of depth
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
        
        hsize_t tileSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        if (2 * tileSize > memoryLimitInPixels) {
            hsize_t memoryRequiredInMb = 2 * tileSize * sizeof(float) / 1024 / 1024;
            std::cerr << "Memory limit too low for tiled rotation. Minimum: " << std::to_string(memoryRequiredInMb) << " mb.";
            std::exit(EXIT_FAILURE);
        }
        hsize_t memoryLimitInTiles = std::floor((float)memoryLimitInPixels / (float)tileSize / 2); //divide by 2 because we have standard and rotated slices
        float widthInTiles = (float)width / (float)TILE_SIZE;
        float heightInTiles = (float)height / (float)TILE_SIZE;
        hsize_t TotalFullTiles = std::ceil(widthInTiles) * std::ceil(heightInTiles);
        hsize_t maxTilesAtATime = std::min(TotalFullTiles, memoryLimitInTiles);
        
        //Allocate memory for main and rotated dataset slices and Z statistics
        float* standardSlice = new float[tileSize * maxTilesAtATime];
        float* rotatedSlice = new float[tileSize * maxTilesAtATime];
        statsZ.createBuffers({TILE_SIZE, TILE_SIZE});
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            PROGRESS("\tStokes " << s << "\t");
            
            hsize_t tileCount(0);
            
            // Calculate excess pixels that won't fit into full tiles
            hsize_t rightStripeWidthInPixels = width % TILE_SIZE;
            hsize_t bottomStripeHeightInPixels = height % TILE_SIZE;
            
            hsize_t mainBlockWidthInTiles = (width - rightStripeWidthInPixels) / TILE_SIZE;
            hsize_t mainBlockHeightInTiles = (height - bottomStripeHeightInPixels) / TILE_SIZE;
            
            hsize_t xTileIncrement = 1;
            hsize_t yTileIncrement = 1;
            
            // Main block of full tiles
            if (!(mainBlockWidthInTiles == 0 && mainBlockHeightInTiles == 0)) {
                if (maxTilesAtATime < mainBlockWidthInTiles) {
                    hsize_t possibleIncrement = 1;
                    while (possibleIncrement < maxTilesAtATime) {
                        possibleIncrement++;
                        if (mainBlockWidthInTiles % possibleIncrement == 0) {
                            xTileIncrement = possibleIncrement;
                        }
                    }
                }
                else {
                    xTileIncrement = mainBlockWidthInTiles;
                    hsize_t possibleIncrement = 1;
                    while (possibleIncrement * xTileIncrement < maxTilesAtATime) {
                        possibleIncrement++;
                        if (mainBlockHeightInTiles % possibleIncrement == 0) {
                            yTileIncrement = possibleIncrement;
                        }
                    }
                }
                DEBUG(std::cout << "+ Processing main block tiles at " << 0 << ", " << 0 << "..." << std::flush;);
                PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");
                ReadRotateWrite(standardSlice, rotatedSlice, s, 0, 0, mainBlockWidthInTiles * TILE_SIZE, mainBlockHeightInTiles * TILE_SIZE,
                    xTileIncrement * TILE_SIZE, yTileIncrement * TILE_SIZE);
                tileCount += xTileIncrement * yTileIncrement;
            }
            
            // Bottom stripe of partial tiles
            if (bottomStripeHeightInPixels > 0 && mainBlockWidthInTiles > 0) {
                DEBUG(std::cout << "+ Processing bottom stripe of tiles at " << 0 << ", " << mainBlockHeightInTiles * TILE_SIZE << "..." << std::flush;);
                yTileIncrement = 1;
                if (maxTilesAtATime > mainBlockWidthInTiles)
                    xTileIncrement = mainBlockWidthInTiles;
                else
                    xTileIncrement = maxTilesAtATime;
                ReadRotateWrite(standardSlice, rotatedSlice, s, 0, mainBlockHeightInTiles * TILE_SIZE, mainBlockWidthInTiles * TILE_SIZE,
                    height, xTileIncrement * TILE_SIZE, bottomStripeHeightInPixels);
                tileCount += mainBlockWidthInTiles;
                PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");
            }
            
            //Right-side stripe of partial tiles
            if (rightStripeWidthInPixels > 0 && mainBlockHeightInTiles > 0) {
                DEBUG(std::cout << "+ Processing right stripe of tiles at " << mainBlockWidthInTiles * TILE_SIZE << ", " << 0 << "..." << std::flush;);
                xTileIncrement = 1;
                if (maxTilesAtATime > mainBlockHeightInTiles)
                    yTileIncrement = mainBlockHeightInTiles;
                else
                    yTileIncrement = maxTilesAtATime;
                ReadRotateWrite(standardSlice, rotatedSlice, s, mainBlockWidthInTiles * TILE_SIZE, 0, width,
                    mainBlockHeightInTiles * TILE_SIZE, rightStripeWidthInPixels, yTileIncrement * TILE_SIZE);
                tileCount += mainBlockHeightInTiles;
                PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");
            }
            
            // Bottom-right partial tile that remains
            DEBUG(std::cout << "+ Processing remainder partial tile in bottom right " << mainBlockWidthInTiles * TILE_SIZE << ", " << mainBlockHeightInTiles * TILE_SIZE << "..." << std::flush;);
            ReadRotateWrite(standardSlice, rotatedSlice, s, mainBlockWidthInTiles * TILE_SIZE, mainBlockHeightInTiles * TILE_SIZE, width,
                height, rightStripeWidthInPixels, bottomStripeHeightInPixels);
            tileCount++;
            PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");
            PROGRESS(std::endl);
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
}
void SmartConverter::ReadRotateWrite(float* standardSliceToRead, float* rotatedSliceToWrite, unsigned int s, hsize_t xStart, hsize_t yStart,
    hsize_t xLimit, hsize_t yLimit, hsize_t xIncrement, hsize_t yIncrement) {
    
    for (hsize_t yOffset = yStart; yOffset < yLimit; yOffset += yIncrement) {
        // If remaining rows are less than the increment, adjust the size of the processed slice
        hsize_t ySize = yOffset + yIncrement > yLimit ? yLimit - yOffset : yIncrement;
        for (hsize_t xOffset = xStart; xOffset < xLimit; xOffset += xIncrement) {
            // If remaining columns are less than the increment, adjust the size of the processed slice
            hsize_t xSize = xOffset + xIncrement > xLimit ? xLimit - xOffset : xIncrement;
            // read tile slice
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            
            auto standardMemDims = trimAxes({1, depth, ySize, xSize}, N);
            auto standardCount = trimAxes({1, depth, ySize, xSize}, N);
            auto standardStart = trimAxes({s, 0, yOffset, xOffset}, N);
            
            readHdf5Data(standardDataSet, standardSliceToRead, standardMemDims, standardCount, standardStart);
            
            // rotate tile slice
            DEBUG(std::cout << " Calculating rotation..." << std::flush;);
            TIMER(timer.start("Rotation"););
            
            hsize_t i;
#pragma omp parallel for default(none) private (i) shared (depth, xSize, ySize, xStart, yStart, standardSliceToRead, rotatedSliceToWrite)
            for (i = 0; i < depth; i++) {
                for (hsize_t j = yStart; j < ySize; j++) {
                    for (hsize_t k = xStart; k < xSize; k++) {
                        auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                        auto& val = standardSliceToRead[sourceIndex];
                
                        // rotation
                        auto destIndex = i + depth * j + (ySize * depth) * k;
                        rotatedSliceToWrite[destIndex] = val;
                    }
                }
            }
            
            // A separate pass over the same slice depth-last
            DEBUG(std::cout << " Calculating Z statistics..." << std::flush;);
            TIMER(timer.start("Z statistics"););
            
            for (hsize_t j = yStart; j < ySize; j++) {
                for (hsize_t k = xStart; k < xSize; k++) {
                    StatsCounter counterZ;
                    auto indexZ = k + xSize * j;
                    
                    for (hsize_t i = 0; i < depth; i++) {
                        auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                        auto& val = standardSliceToRead[sourceIndex];
                        
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

            // write tile slice
            DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
            TIMER(timer.start("Write"););
            
            auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
            auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
            auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);
            
            writeHdf5Data(swizzledDataSet, rotatedSliceToWrite, swizzledMemDims, swizzledCount, swizzledStart);
            
            DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
            // write Z statistics
            statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
        }
    }
}
