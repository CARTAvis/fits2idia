#include "Converter.h"

SlowConverter::SlowConverter(std::string inputFileName, std::string outputFileName) : Converter(inputFileName, outputFileName) {
    MipMap::initialise(mipMaps, N, width, height, 1);
    timer = Timer(stokes * depth * height * width, true);
}

void SlowConverter::copy() {
    // Allocate one channel at a time, and no swizzled data
    hsize_t cubeSize = height * width;
    allocate(cubeSize);
    
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
        std::cout << "Processing Stokes " << s << " dataset... " << std::endl;
        
        for (hsize_t c = 0; c < depth; c++) {
            // read one channel
            long fpixel[] = {1, 1, (long)c + 1, s + 1};
            readFits(fpixel, cubeSize);
            
            // Write the standard dataset
            if (N == 2) {
                start = {0, 0};
            } else if (N == 3) {
                start = {c, 0, 0};
            } else if (N == 4) {
                start = {s, c, 0, 0};
            }

            timer.write.start();
            sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
            standardDataSet.write(standardCube, H5::PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
            timer.write.stop();
            
            timer.process1.start();
            
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
                                        
                    if (!std::isnan(val)) {
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
            
            // TODO: see if this can be done in stats
            if (statsXY.nanCounts[indexXY] == (height * width)) {
                statsXY.minVals[indexXY] = NAN;
                statsXY.maxVals[indexXY] = NAN;
            }
            
            // Accumulate XYZ statistics
            
            if (depth > 1) {
                if (!std::isnan(statsXY.sums[indexXY])) {
                    statsXYZ.sums[s] += statsXY.sums[indexXY];
                    statsXYZ.sumsSq[s] += statsXY.sumsSq[indexXY];
                    statsXYZ.minVals[s] = fmin(statsXYZ.minVals[s], statsXY.minVals[indexXY]);
                    statsXYZ.maxVals[s] = fmax(statsXYZ.maxVals[s], statsXY.maxVals[indexXY]);
                }
                statsXYZ.nanCounts[s] += statsXY.nanCounts[indexXY];
            }
            
            // Final mipmap calculation
            for (auto& mipMap : mipMaps) {
                mipMap.calculate();
            }
            
            timer.process1.stop();
            
            // Write the mipmaps
            
            timer.write.start();
        
            for (auto& mipMap : mipMaps) {
                // Start at current Stokes and channel
                mipMap.write(s, c);
            }
            
            timer.write.stop();
            timer.process1.start();
            
            // Reset mipmaps before next channel
            for (auto& mipMap : mipMaps) {
                mipMap.reset();
            }
            
            timer.process1.stop();
            
        } // end of first channel loop
        
        if (depth > 1) {
            timer.process1.start();
            
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
            
            if (statsXYZ.nanCounts[s] == depth * height * width) {
                statsXYZ.minVals[s] = NAN;
                statsXYZ.maxVals[s] = NAN;
            }
            
            timer.process1.stop();
        }
        
        // XY and XYZ histograms
        // We need a second pass over all channels because we need cube min and max (and channel min and max per channel)
        // We do the second pass backwards to take advantage of caching
        
        timer.process2.start();
        
        bool cubeHist(0);
        double cubeMin;
        double cubeMax;
        double cubeRange;
        
        if (depth > 1) {
            cubeMin = statsXYZ.minVals[s];
            cubeMax = statsXYZ.maxVals[s];
            cubeRange = cubeMax - cubeMin;
            cubeHist = !std::isnan(cubeMin) && !std::isnan(cubeMax) && cubeRange > 0;
        }
        
        for (hsize_t c = depth; c-- > 0; ) {                
            auto indexXY = s * depth + c;
                            
            double chanMin = statsXY.minVals[indexXY];
            double chanMax = statsXY.maxVals[indexXY];
            double chanRange = chanMax - chanMin;
            bool chanHist(!std::isnan(chanMin) && !std::isnan(chanMax) && chanRange > 0);
            
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
            long fpixel[] = {1, 1, (long)c + 1, s + 1};
            
            timer.process2.stop();
            timer.read.start();
            
            readFits(fpixel, cubeSize);
            
            timer.read.stop();
            timer.process2.start();

            for (auto p = 0; p < width * height; p++) {
                auto val = standardCube[p];
                    if (!std::isnan(val)) {
                        channelHistogramFunc(val);
                        cubeHistogramFunc(val);
                    }
            } // end of XY loop
        } // end of second channel loop (XY and XYZ histograms)
        
        timer.process2.stop();
        
    } // end of stokes
            
    // Swizzle
    if (depth > 1) {
        std::cout << "Performing slow, memory-saving rotation." << std::endl;
        
        hsize_t sliceSize;
    
        if (N == 3) {
            sliceSize = depth * TILE_SIZE * TILE_SIZE;
        } else if (N == 4) {
            sliceSize = stokes * depth * TILE_SIZE * TILE_SIZE;
        }
        
        std::cout << "Allocating " << sliceSize * 4 * 2 * 1e-9 << " GB of memory for main and rotated dataset slices... " << std::endl;
        timer.alloc.start();
        
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];

        timer.alloc.stop();
        
        auto standardDataSpace = standardDataSet.getSpace();
        auto swizzledDataSpace = swizzledDataSet.getSpace();
        
        for (unsigned int s = 0; s < stokes; s++) {
            for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                    hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
                    hsize_t ySize = std::min(TILE_SIZE, height - yOffset);
                    
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
                    timer.read.start();
                    
                    standardDataSpace.selectHyperslab(H5S_SELECT_SET, standardCount.data(), standardOffset.data());
                    standardDataSet.read(standardSlice, H5::PredType::NATIVE_FLOAT, standardMemspace, standardDataSpace);
                    
                    timer.read.stop();
                    timer.process3.start();
                    
                    // rotate tile slice
                    for (auto i = 0; i < depth; i++) {
                        for (auto j = 0; j < ySize; j++) {
                            for (auto k = 0; k < xSize; k++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto destIndex = i + depth * j + (ySize * depth) * k;
                                rotatedSlice[destIndex] = standardSlice[sourceIndex];
                            }
                        }
                    }
                    
                    timer.process3.stop();
                    timer.write.start();
                            
                    // write tile slice
                    swizzledDataSpace.selectHyperslab(H5S_SELECT_SET, swizzledCount.data(), swizzledOffset.data());
                    swizzledDataSet.write(rotatedSlice, H5::PredType::NATIVE_FLOAT, swizzledMemspace, swizzledDataSpace);
                    
                    timer.write.stop();
                }
            }
        }
        
        std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
}
