#include "Converter.h"

SlowConverter::SlowConverter(std::string inputFileName, std::string outputFileName, bool progress) : Converter(inputFileName, outputFileName, progress) {}

void SlowConverter::reportMemoryUsage() {
    std::unordered_map<std::string, hsize_t> sizes;

    sizes["Main dataset"] = height * width * sizeof(float);
    sizes["Mipmaps"] = MipMaps::size(standardDims, {1, height, width});
    sizes["XY stats"] = Stats::size({depth}, numBins);
    
    if (depth > 1) {
        sizes["Rotation"] = 2 * product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        sizes["Z stats"] = Stats::size({TILE_SIZE, TILE_SIZE});
    }
    
    hsize_t total(0);
    
    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;
    
    for (auto& kv : sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
        total += kv.second;
    }
    
    std::string note = "";
    if (depth > 1) {
        total -= std::min(sizes["Main dataset"], sizes["Rotation"] + sizes["Z stats"]);
        note = " (Main dataset and slices for rotation and Z statistics are not allocated at the same time.)";
    }

    std::cout << "TOTAL:\t" << total * 1e-9 << "GB" << note << std::endl;
}

void SlowConverter::copyAndCalculate() {
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

    
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        
        for (hsize_t c = 0; c < depth; c++) {
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

            auto indexXY = c;
            std::function<void(float)> accumulate;
            
            auto lazy_accumulate = [&] (float val) {
                statsXY.accumulateFiniteLazy(indexXY, val);
            };
            
            auto first_accumulate = [&] (float val) {
                statsXY.accumulateFiniteLazyFirst(indexXY, val);
                accumulate = lazy_accumulate;
            };
            
            accumulate = first_accumulate;
            
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
                        statsXY.accumulateNonFinite(indexXY);
                    }
                }
            } // end of XY loop
            
            // Final correction of XY min and max
            DEBUG(std::cout << " Final XY stats..." << std::flush;);
            statsXY.finalMinMax(indexXY, height * width);
            
            // Accumulate XYZ statistics
            if (depth > 1) {
                DEBUG(std::cout << " Accumulating XYZ stats..." << std::flush;);
                statsXYZ.accumulateStats(statsXY, 0, indexXY);
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
        
        if (depth > 1) {
            // Final correction of XYZ min and max
            DEBUG(std::cout << " Final XYZ stats..." << std::flush;);
            TIMER(timer.start(timerLabelStatsMipmaps););
            statsXYZ.finalMinMax(0, depth * height * width);
        }
        
        // XY and XYZ histograms
        // We need a second pass over all channels because we need cube min and max (and channel min and max per channel)
        // We do the second pass backwards to take advantage of caching
        DEBUG(std::cout << " Histograms..." << std::endl;);
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
        
        DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);
        
        for (hsize_t c = depth; c-- > 0; ) {
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
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
            
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube);

            DEBUG(std::cout << " Calculating histogram(s)..." << std::endl;);
            TIMER(timer.start("Histograms"););
            
            for (hsize_t p = 0; p < width * height; p++) {
                auto& val = standardCube[p];
                    if (std::isfinite(val)) {
                        channelHistogramFunc(val);
                        cubeHistogramFunc(val);
                    }
            } // end of XY loop
        } // end of second channel loop (XY and XYZ histograms)
        
        // Write the statistics
        TIMER(timer.start("Write"););
                
        statsXY.write({1, depth}, {s, 0});
        
        if (depth > 1) {
            statsXYZ.write({1}, {s});
        }
        
        // Clear the stats before the next Stokes
        TIMER(timer.start(timerLabelStatsMipmaps););
        
        statsXY.resetBuffers();
        
        TIMER(timer.start(timerLabelStatsMipmaps););
        
        if (depth > 1) {
            statsXYZ.resetBuffers();
        }
    
    } // end of stokes
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
            
    // Swizzle
    if (depth > 1) {
        DEBUG(std::cout << "Performing tiled rotation." << std::endl;);
        TIMER(timer.start("Allocate"););
        
        hsize_t sliceSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];
        
        statsZ.createBuffers({TILE_SIZE, TILE_SIZE});
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                    hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
                    hsize_t ySize = std::min(TILE_SIZE, height - yOffset);
                    
                    DEBUG(std::cout << "+ Processing tile slice at " << xOffset << ", " << yOffset << "..." << std::flush;);
                    
                    // read tile slice
                    DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                    TIMER(timer.start("Read"););
                    
                    auto standardMemDims = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardCount = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardStart = trimAxes({s, 0, yOffset, xOffset}, N);
                    
                    readHdf5Data(standardDataSet, standardSlice, standardMemDims, standardCount, standardStart);
                    
                    // rotate tile slice
                    DEBUG(std::cout << " Calculating rotation and Z statistics..." << std::flush;);
                    TIMER(timer.start("Rotation and Z statistics"););
                    
                    for (hsize_t i = 0; i < depth; i++) {
                        for (hsize_t j = 0; j < ySize; j++) {
                            for (hsize_t k = 0; k < xSize; k++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto& val = standardSlice[sourceIndex];
                                
                                // rotation
                                auto destIndex = i + depth * j + (ySize * depth) * k;
                                rotatedSlice[destIndex] = val;
                                
                                // Accumulate Z statistics                                
                                auto indexZ = k + xSize * j;
                                
                                if (std::isfinite(val)) {
                                    // Not lazy; too much risk of encountering an ascending / descending sequence.
                                    statsZ.accumulateFinite(indexZ, val);
                                } else {
                                    statsZ.accumulateNonFinite(indexZ);
                                }
                            }
                        }
                    }
                    
                    DEBUG(std::cout << " Final Z stats..." << std::flush;);
                    // A final pass over all XY to fix the Z stats NaNs
                    
                    for (hsize_t p = 0; p < xSize * ySize; p++) {
                        auto& indexZ = p;
                        
                        statsZ.finalMinMax(indexZ, depth);
                    }
                    
                    // write tile slice
                    DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                    TIMER(timer.start("Write"););
                    
                    auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);
                    
                    writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                    
                    DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
                    // write Z statistics
                    statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
                    
                    // reset Z statistics buffers before the next slice
                    statsZ.resetBuffers();
                }
            }
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
}
