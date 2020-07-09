#include "Converter.h"

SlowConverter::SlowConverter(std::string inputFileName, std::string outputFileName) : Converter(inputFileName, outputFileName) {
    TIMER(timer.start("Mipmaps"););
    MipMap::initialise(mipMaps, N, width, height, 1);
}

void SlowConverter::reportMemoryUsage() {
    std::unordered_map<std::string, hsize_t> sizes;

    sizes["Main dataset"] = height * width * sizeof(float);
    sizes["Mipmaps"] = MipMap::size(width, height, 1);
    sizes["XY stats"] = Stats::size(dims.statsXY);
    
    if (depth > 1) {
        if (N == 3) {
            sizes["Rotation"] = 2 * depth * TILE_SIZE * TILE_SIZE;
        } else if (N == 4) {
            sizes["Rotation"] = 2 * stokes * depth * TILE_SIZE * TILE_SIZE;
        }
        sizes["XYZ stats"] = Stats::size(dims.statsXYZ);
        sizes["Z stats"] = Stats::size(dims.statsZ);
    }
    
    hsize_t total(0);
    
    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;
    
    for (auto& kv : sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
        total += kv.second;
    }
    
    std::string note = "";
    if (depth > 1) {
        total -= std::min(sizes["Main dataset"], sizes["Rotation"]);
        note = " (Main dataset and slices for rotation are not allocated at the same time.)";
    }

    std::cout << "TOTAL:\t" << total * 1e-9 << "GB" << note << std::endl;
}

void SlowConverter::copyAndCalculate() {
    // Allocate one channel at a time, and no swizzled data
    hsize_t cubeSize = height * width;
    TIMER(timer.start("Allocate"););
    standardCube = new float[cubeSize];
    
    // TODO these sizes will be different when we don't store all the stats at once
    statsXY.createBuffers(dims.statsXY.statsSize, dims.statsXY.histSize);
    
    if (depth > 1) {
        statsZ.createBuffers(dims.statsZ.statsSize);
        statsXYZ.createBuffers(dims.statsXYZ.statsSize, dims.statsXYZ.histSize, dims.statsXYZ.partialHistSize);
    }
    
    auto sliceDataSpace = standardDataSet.getSpace();
                        
    std::vector<hsize_t> count;
    std::vector<hsize_t> start;
    
    if (N == 2) {
        count = {height, width};
    } else if (N == 3) {
        count = {1, height, width};
    } else if (N == 4) {
        count = {1, 1, height, width};
    }
    
    hsize_t memDims[] = {height, width};
    H5::DataSpace memspace(2, memDims);
    
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        
        for (hsize_t c = 0; c < depth; c++) {
            // read one channel
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube);
            
            // Write the standard dataset
            if (N == 2) {
                start = {0, 0};
            } else if (N == 3) {
                start = {c, 0, 0};
            } else if (N == 4) {
                start = {s, c, 0, 0};
            }
            
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
            standardDataSet.write(standardCube, H5::PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
            
            TIMER(timer.start("Statistics and mipmaps"););
            
            DEBUG(std::cout << " Accumulating XY " << (depth > 1 ? "and Z " : "") << "stats and mipmaps..." << std::flush;);
            auto indexXY = s * depth + c;
            
            std::function<void(float)> minmax;
            
            auto lazy_minmax = [&] (float val) {
                if (val < statsXY.minVals[indexXY]) {
                    statsXY.minVals[indexXY] = val;
                } else if (val > statsXY.maxVals[indexXY]) {
                    statsXY.maxVals[indexXY] = val;
                }
            };
            
            auto first_minmax = [&] (float val) {
                statsXY.minVals[indexXY] = val;
                statsXY.maxVals[indexXY] = val;
                minmax = lazy_minmax;
            };
            
            minmax = first_minmax;
            
            for (auto y = 0; y < height; y++) {
                for (auto x = 0; x < width; x++) {
                    auto pos = y * width + x; // relative to channel slice
                    auto indexZ = s * width * height + pos; // relative to whole image
                    
                    auto val = standardCube[pos];
                                        
                    if (std::isfinite(val)) {
                        // XY statistics
                        // TODO: check if indexing stats arrays inside loop is slow or is optimised away
                        minmax(val);
                        statsXY.sums[indexXY] += val;
                        statsXY.sumsSq[indexXY] += val * val;
                        
                        if (depth > 1) {
                            // Accumulate Z statistics
                            // Not replacing this with if/else; too much risk of encountering an ascending / descending sequence.
                            statsZ.minVals[indexZ] = fmin(statsZ.minVals[indexZ], val);
                            statsZ.maxVals[indexZ] = fmax(statsZ.maxVals[indexZ], val);
                            statsZ.sums[indexZ] += val;
                            statsZ.sumsSq[indexZ] += val * val;
                        }
                        
                        // Accumulate mipmaps
                        for (auto& mipMap : mipMaps) {
                            mipMap.accumulate(val, x, y, 0);
                        }
                        
                    } else {
                        statsXY.nanCounts[indexXY] += 1;
                        if (depth > 1) {
                            statsZ.nanCounts[indexZ] += 1;
                        }
                    }
                }
            } // end of XY loop
            
            DEBUG(std::cout << " Final XY stats..." << std::flush;);
            // TODO: see if this can be done in stats
            if (statsXY.nanCounts[indexXY] == (height * width)) {
                statsXY.minVals[indexXY] = NAN;
                statsXY.maxVals[indexXY] = NAN;
            }
            
            // Accumulate XYZ statistics
            if (depth > 1) {
                DEBUG(std::cout << " Accumulating XYZ stats..." << std::flush;);
                if (std::isfinite(statsXY.maxVals[indexXY])) {
                    statsXYZ.sums[s] += statsXY.sums[indexXY];
                    statsXYZ.sumsSq[s] += statsXY.sumsSq[indexXY];
                    statsXYZ.minVals[s] = fmin(statsXYZ.minVals[s], statsXY.minVals[indexXY]);
                    statsXYZ.maxVals[s] = fmax(statsXYZ.maxVals[s], statsXY.maxVals[indexXY]);
                }
                statsXYZ.nanCounts[s] += statsXY.nanCounts[indexXY];
            }
            
            // Final mipmap calculation
            DEBUG(std::cout << " Final mipmaps..." << std::flush;);
            for (auto& mipMap : mipMaps) {
                mipMap.calculate();
            }
            
            // Write the mipmaps
            
            TIMER(timer.start("Write"););
            DEBUG(std::cout << " Writing mipmaps..." << std::flush;);
            for (auto& mipMap : mipMaps) {
                // Start at current Stokes and channel
                mipMap.write(s, c);
            }
            
            TIMER(timer.start("Statistics and mipmaps"););
            
            // Reset mipmaps before next channel
            DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
            for (auto& mipMap : mipMaps) {
                mipMap.reset();
            }
            
        } // end of first channel loop
        
        if (depth > 1) {
            TIMER(timer.start("Statistics and mipmaps"););
            DEBUG(std::cout << "+ Final Z stats..." << std::flush;);
            
            // A final pass over all XY to fix the Z stats NaNs
            for (auto p = 0; p < width * height; p++) {
                auto indexZ = s * width * height + p;
                                        
                // TODO can we do this in stats?
                if (statsZ.nanCounts[indexZ] == depth) {
                    statsZ.minVals[indexZ] = NAN;
                    statsZ.maxVals[indexZ] = NAN;
                }
            }
            
            // A final correction of the XYZ NaNs
            DEBUG(std::cout << " Final XYZ stats..." << std::flush;);
            
            if (statsXYZ.nanCounts[s] == depth * height * width) {
                statsXYZ.minVals[s] = NAN;
                statsXYZ.maxVals[s] = NAN;
            }
        }
        
        // XY and XYZ histograms
        // We need a second pass over all channels because we need cube min and max (and channel min and max per channel)
        // We do the second pass backwards to take advantage of caching
        DEBUG(std::cout << " Histograms..." << std::endl;);
        TIMER(timer.start("Histograms"););
        
        bool cubeHist(0);
        double cubeMin;
        double cubeMax;
        double cubeRange;
        
        if (depth > 1) {
            cubeMin = statsXYZ.minVals[s];
            cubeMax = statsXYZ.maxVals[s];
            cubeRange = cubeMax - cubeMin;
            cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
        }
        
        DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);
        
        for (hsize_t c = depth; c-- > 0; ) {
            DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
            auto indexXY = s * depth + c;
                            
            double chanMin = statsXY.minVals[indexXY];
            double chanMax = statsXY.maxVals[indexXY];
            double chanRange = chanMax - chanMin;
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
            
            DEBUG(std::cout << " Will " << (chanHist ? "" : "not ") << "calculate channel histogram." << std::flush;);
            
            std::function<void(float)> channelHistogramFunc;
            std::function<void(float)> cubeHistogramFunc;
            
            auto doChannelHistogram = [&] (float val) {
                // XY Histogram
                int binIndex = std::min(numBinsXY - 1, (int)(numBinsXY * (val - chanMin) / chanRange));
                statsXY.histograms[s * depth * numBinsXY + c * numBinsXY + binIndex]++;
            };
            
            auto doCubeHistogram = [&] (float val) {
                // XYZ histogram
                int binIndex = std::min(numBinsXYZ - 1, (int)(numBinsXYZ * (val - cubeMin) / cubeRange));
                statsXYZ.histograms[s * numBinsXYZ + binIndex]++;
            };
            
            auto doNothing = [&] (float val) {};
            
            channelHistogramFunc = doChannelHistogram;
            cubeHistogramFunc = doCubeHistogram;
            
            if (!chanHist) {
                channelHistogramFunc = doNothing;
            }
            
            if (!cubeHist) {
                cubeHistogramFunc = doNothing;
            }
            
            if (!chanHist && !cubeHist) {
                continue;
            }
            
            // read one channel
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            readFitsData(inputFilePtr, c, s, cubeSize, standardCube);
            
            TIMER(timer.start("Histograms"););

            DEBUG(std::cout << " Calculating histogram(s)..." << std::endl;);
            for (auto p = 0; p < width * height; p++) {
                auto val = standardCube[p];
                    if (std::isfinite(val)) {
                        channelHistogramFunc(val);
                        cubeHistogramFunc(val);
                    }
            } // end of XY loop
        } // end of second channel loop (XY and XYZ histograms)
    
    } // end of stokes
    
    // Free memory
    TIMER(timer.start("Free"););
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    delete[] standardCube;
            
    // Swizzle
    if (depth > 1) {
        DEBUG(std::cout << "Performing tiled rotation." << std::endl;);
        
        hsize_t sliceSize;
    
        if (N == 3) {
            sliceSize = depth * TILE_SIZE * TILE_SIZE;
        } else if (N == 4) {
            sliceSize = stokes * depth * TILE_SIZE * TILE_SIZE;
        }
        
        TIMER(timer.start("Allocate"););
        
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];

        auto standardDataSpace = standardDataSet.getSpace();
        auto swizzledDataSpace = swizzledDataSet.getSpace();
        
        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                    hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
                    hsize_t ySize = std::min(TILE_SIZE, height - yOffset);
                    
                    DEBUG(std::cout << "+ Processing tile slice at " << xOffset << ", " << yOffset << "..." << std::flush;);
                    
                    std::vector<hsize_t> standardMemDims;
                    std::vector<hsize_t> swizzledMemDims;
                    std::vector<hsize_t> standardOffset;
                    std::vector<hsize_t> standardCount;
                    std::vector<hsize_t> swizzledOffset;
                    std::vector<hsize_t> swizzledCount;
                    
                    if (N == 3) {
                        standardMemDims = {depth, ySize, xSize};
                        swizzledMemDims = {xSize, ySize, depth};
                        standardOffset = {0, yOffset, xOffset};
                        standardCount = {depth, ySize, xSize};
                        swizzledOffset = {xOffset, yOffset, 0};
                        swizzledCount = {xSize, ySize, depth};
                    } else if (N == 4) {
                        standardMemDims = {1, depth, ySize, xSize};
                        swizzledMemDims = {1, xSize, ySize, depth};
                        standardOffset = {s, 0, yOffset, xOffset};
                        standardCount = {1, depth, ySize, xSize};
                        swizzledOffset = {s, xOffset, yOffset, 0};
                        swizzledCount = {1, xSize, ySize, depth};
                    }
                    
                    H5::DataSpace standardMemspace(standardMemDims.size(), standardMemDims.data());
                    H5::DataSpace swizzledMemspace(swizzledMemDims.size(), swizzledMemDims.data());
                    
                    // read tile slice
                    DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                    TIMER(timer.start("Read"););
                    
                    standardDataSpace.selectHyperslab(H5S_SELECT_SET, standardCount.data(), standardOffset.data());
                    standardDataSet.read(standardSlice, H5::PredType::NATIVE_FLOAT, standardMemspace, standardDataSpace);
                    
                    TIMER(timer.start("Rotation"););
                    
                    // rotate tile slice
                    DEBUG(std::cout << " Calculating rotation..." << std::flush;);
                    for (auto i = 0; i < depth; i++) {
                        for (auto j = 0; j < ySize; j++) {
                            for (auto k = 0; k < xSize; k++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto destIndex = i + depth * j + (ySize * depth) * k;
                                rotatedSlice[destIndex] = standardSlice[sourceIndex];
                            }
                        }
                    }
                    
                    TIMER(timer.start("Write"););
                            
                    // write tile slice
                    DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                    swizzledDataSpace.selectHyperslab(H5S_SELECT_SET, swizzledCount.data(), swizzledOffset.data());
                    swizzledDataSet.write(rotatedSlice, H5::PredType::NATIVE_FLOAT, swizzledMemspace, swizzledDataSpace);
                }
            }
        }
        
        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
}
