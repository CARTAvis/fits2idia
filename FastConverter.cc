#include "Converter.h"

FastConverter::FastConverter(std::string inputFileName, std::string outputFileName) : Converter(inputFileName, outputFileName) {
    timer = Timer(stokes * depth * height * width, false);
    timer.start("Mipmaps");
    MipMap::initialise(mipMaps, N, width, height, depth);
}

void FastConverter::copy() {
    // TODO fix reporting of total memory allocation -- size * 4 * 1e-9 GB
    hsize_t cubeSize = depth * height * width;
    timer.start("Allocate");
    standardCube = new float[cubeSize];
    
    statsXY = Stats(dims.statsXY);
    
    if (depth > 1) {
        statsZ = Stats(dims.statsZ);
        statsXYZ = Stats(dims.statsXYZ);
    }

    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        // Read data into memory space
        timer.start("Read");
        std::cout << "Reading Stokes " << currentStokes << " dataset... " << std::endl;
        readFits(0, currentStokes, cubeSize);
        
        // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
        if (depth > 1) {
            timer.start("Allocate");
            rotatedCube = new float[cubeSize];
        }

        std::cout << "Processing Stokes " << currentStokes << " dataset..." << std::endl;
        
        std::string process_label = depth > 1 ? "XY statistics and rotation" : "XY statistics";
        
        timer.start(process_label);
        
        std::cout << " * " << process_label <<  "... " << std::endl;

        // First loop calculates stats for each XY slice and rotates the dataset
#pragma omp parallel for
        for (auto i = 0; i < depth; i++) {
            float minVal = std::numeric_limits<float>::max();
            float maxVal = -std::numeric_limits<float>::max();
            double sum = 0;
            double sumSq = 0;
            int64_t nanCount = 0;
            
            std::function<void(float)> minmax;
            
            auto lazy_minmax = [&] (float val) {
                if (val < minVal) {
                    minVal = val;
                } else if (val > maxVal) {
                    maxVal = val;
                }
            };
            
            auto first_minmax = [&] (float val) {
                minVal = val;
                maxVal = val;
                minmax = lazy_minmax;
            };
            
            minmax = first_minmax;
            
            for (auto j = 0; j < height; j++) {
                for (auto k = 0; k < width; k++) {
                    auto sourceIndex = k + width * j + (height * width) * i;
                    auto destIndex = i + depth * j + (height * depth) * k;
                    auto val = standardCube[sourceIndex];
                    
                    if (depth > 1) {
                        rotatedCube[destIndex] = val;
                    }
                    
                    if (!std::isnan(val)) {
                        minmax(val);
                        sum += val;
                        sumSq += val * val;
                    } else {
                        nanCount += 1;
                    }
                }
            }

            auto indexXY = currentStokes * depth + i;
            
            statsXY.nanCounts[indexXY] = nanCount;
            statsXY.sums[indexXY] = sum;
            statsXY.sumsSq[indexXY] = sumSq;
            
            if (nanCount != (height * width)) {
                statsXY.minVals[indexXY] = minVal;
                statsXY.maxVals[indexXY] = maxVal;
            } else {
                statsXY.minVals[indexXY] = NAN;
                statsXY.maxVals[indexXY] = NAN;
            }
        }
        
        double xyzMin;
        double xyzMax;

        if (depth > 1) {
            timer.start("XYZ and Z statistics");
            
            // Consolidate XY stats into XYZ stats
            std::cout << " * XYZ statistics... " << std::endl;
            double xyzSum = 0;
            double xyzSumSq = 0;
            int64_t xyzNanCount = 0;
            xyzMin = statsXY.minVals[currentStokes * depth];
            xyzMax = statsXY.maxVals[currentStokes * depth];

            for (auto i = 0; i < depth; i++) {
                auto indexXY = currentStokes * depth + i;
                auto sum = statsXY.sums[indexXY];
                if (!std::isnan(sum)) {
                    xyzSum += sum;
                    xyzSumSq += statsXY.sumsSq[indexXY];
                    xyzMin = fmin(xyzMin, statsXY.minVals[indexXY]);
                    xyzMax = fmax(xyzMax, statsXY.maxVals[indexXY]);
                }
                xyzNanCount += statsXY.nanCounts[indexXY];
            }

            statsXYZ.nanCounts[currentStokes] = xyzNanCount;
            statsXYZ.sums[currentStokes] = xyzSum;
            statsXYZ.sumsSq[currentStokes] = xyzSumSq;
            
            if (xyzNanCount != depth * height * width) {
                statsXYZ.minVals[currentStokes] = xyzMin;
                statsXYZ.maxVals[currentStokes] = xyzMax;
            } else {
                statsXYZ.minVals[currentStokes] = NAN;
                statsXYZ.maxVals[currentStokes] = NAN;
            }

            std::cout << " * Z statistics... " << std::endl;
            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
            
#pragma omp parallel for
            for (auto j = 0; j < height; j++) {
                for (auto k = 0; k < width; k++) {
                    float minVal = std::numeric_limits<float>::max();
                    float maxVal = -std::numeric_limits<float>::max();
                    double sum = 0;
                    double sumSq = 0;
                    int64_t nanCount = 0;
                    
                    for (auto i = 0; i < depth; i++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto val = standardCube[sourceIndex];

                        if (!std::isnan(val)) {
                            // Not replacing this with if/else; too much risk of encountering an ascending / descending sequence.
                            minVal = std::min(minVal, val);
                            maxVal = std::max(maxVal, val);
                            sum += val;
                            sumSq += val * val;
                        } else {
                            nanCount += 1;
                        }
                    }
                    
                    auto indexZ = currentStokes * width * height + k + j * width;
                    
                    statsZ.nanCounts[indexZ] = nanCount;
                    statsZ.sums[indexZ] = sum;
                    statsZ.sumsSq[indexZ] = sumSq;
                    
                    if (nanCount != depth) {
                        statsZ.minVals[indexZ] = minVal;
                        statsZ.maxVals[indexZ] = maxVal;
                    } else {
                        statsZ.minVals[indexZ] = NAN;
                        statsZ.maxVals[indexZ] = NAN;
                    }
                }
            }
            
            
        }
        
        std::cout << " * Histograms... " << std::endl;

        // Third loop handles histograms
        
        timer.start("Histograms");
        
        double cubeMin;
        double cubeMax;
        double cubeRange;
                    
        if (depth > 1) {
            cubeMin = xyzMin;
            cubeMax = xyzMax;
            cubeRange = cubeMax - cubeMin;
        }

#pragma omp parallel for
        for (auto i = 0; i < depth; i++) {
            auto indexXY = currentStokes * depth + i;
            double sliceMin = statsXY.minVals[indexXY];
            double sliceMax = statsXY.maxVals[indexXY];
            double range = sliceMax - sliceMin;
            
            std::function<void(float)> channelHistogramFunc;
            std::function<void(float)> cubeHistogramFunc;
            
            auto doChannelHistogram = [&] (float val) {
                // XY Histogram
                int binIndex = std::min(numBinsXY - 1, (int)(numBinsXY * (val - sliceMin) / range));
                statsXY.histograms[currentStokes * depth * numBinsXY + i * numBinsXY + binIndex]++;
            };
            
            auto doCubeHistogram = [&] (float val) {
                // XYZ Partial histogram
                int binIndexXYZ = std::min(numBinsXYZ - 1, (int)(numBinsXYZ * (val - cubeMin) / cubeRange));
                statsXYZ.partialHistograms[currentStokes * depth * numBinsXYZ + i * numBinsXYZ + binIndexXYZ]++;
            };
            
            auto doNothing = [&] (float val) {};
            
            channelHistogramFunc = doChannelHistogram;
            cubeHistogramFunc = doCubeHistogram;
            bool chanHist(true);
            bool cubeHist(true);
            
            if (std::isnan(sliceMin) || std::isnan(sliceMax) || range == 0) {
                channelHistogramFunc = doNothing;
                chanHist = false;
            }
            
            if (depth <= 1) {
                cubeHistogramFunc = doNothing;
                cubeHist = false;
            }
            
            if (!chanHist && !cubeHist) {
                continue; // skip the loop entirely
            }

            for (auto j = 0; j < width * height; j++) {
                auto val = standardCube[i * width * height + j];

                if (!std::isnan(val)) {
                    channelHistogramFunc(val);
                    cubeHistogramFunc(val);
                }
            } // end of XY loop
        } // end of parallel Z loop
        
        if (depth > 1) {
            // Consolidate partial XYZ histograms into final histogram
            for (auto i = 0; i < depth; i++) {
                for (auto j = 0; j < numBinsXYZ; j++) {
                    statsXYZ.histograms[currentStokes * numBinsXYZ + j] +=
                        statsXYZ.partialHistograms[currentStokes * depth * numBinsXYZ + i * numBinsXYZ + j];
                }
            }
        }

        

        std::cout << "Writing Stokes " << currentStokes << " dataset... " << std::endl;
        
        timer.start("Write");
                    
        std::vector<hsize_t> count;
        std::vector<hsize_t> start;
        
        if (N == 2) {
            count = {height, width};
            start = {0, 0};
        } else if (N == 3) {
            count = {depth, height, width};
            start = {0, 0, 0};
        } else if (N == 4) {
            count = {1, depth, height, width};
            start = {currentStokes, 0, 0, 0};
        }
        
        hsize_t memDims[] = {depth, height, width};
        H5::DataSpace memspace(3, memDims);

        auto sliceDataSpace = standardDataSet.getSpace();
        sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

        standardDataSet.write(standardCube, H5::PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
        
        if (depth > 1) {
            // This all technically worked if we reused the standard filespace and memspace
            // But it's probably not a good idea to rely on two incorrect values cancelling each other out
            std::vector<hsize_t> swizzledCount;
            
            if (N == 3) {
                swizzledCount = {width, height, depth};
            } else if (N == 4) {
                swizzledCount = {1, width, height, depth};
            }
            
            hsize_t swizzledMemDims[] = {width, height, depth};
            H5::DataSpace swizzledMemspace(3, swizzledMemDims);
            
            auto swizzledDataSpace = swizzledDataSet.getSpace();
            swizzledDataSpace.selectHyperslab(H5S_SELECT_SET, swizzledCount.data(), start.data());
            
            swizzledDataSet.write(rotatedCube, H5::PredType::NATIVE_FLOAT, swizzledMemspace, swizzledDataSpace);
        }

        // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
        
        if (depth > 1) {
            timer.start("Free");
            std::cout << "Freeing memory from rotated dataset... " << std::endl;
            delete[] rotatedCube;
        }
        
        // Fourth loop handles mipmaps
        
        // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
        
        std::cout << " * Mipmaps... " << std::endl;

        timer.start("Mipmaps");
        
#pragma omp parallel for
        for (auto c = 0; c < depth; c++) {
            for (auto y = 0; y < height; y++) {
                for (auto x = 0; x < width; x++) {
                    auto sourceIndex = x + width * y + (height * width) * c;
                    auto val = standardCube[sourceIndex];
                    if (!std::isnan(val)) {
                        for (auto& mipMap : mipMaps) {
                            mipMap.accumulate(val, x, y, c);
                        }
                    }
                }
            }
        } // end of mipmap loop
        
        // Final mipmap calculation
        
        for (auto& mipMap : mipMaps) {
            mipMap.calculate();
        }
        
        
        timer.start("Write");
        
        // Write the mipmaps
        
        for (auto& mipMap : mipMaps) {
            // Start at current Stokes and channel 0
            mipMap.write(currentStokes, 0);
        }
        
        
        
        // Clear the mipmaps before the next Stokes
        
        timer.start("Mipmaps");
        
        for (auto& mipMap : mipMaps) {
            mipMap.reset();
        }
        
        
    } // end of Stokes loop
}
