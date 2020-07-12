#include "Converter.h"

// TODO do we need these?
FastConverter::FastConverter(std::string inputFileName, std::string outputFileName) : Converter(inputFileName, outputFileName) {}

void FastConverter::reportMemoryUsage() {
    std::unordered_map<std::string, hsize_t> sizes;

    sizes["Main dataset"] = depth * height * width * sizeof(float);
    sizes["Mipmaps"] = MipMaps::size(standardDims, {depth, height, width});
    sizes["XY stats"] = Stats::size({depth}, numBins);
    
    if (depth > 1) {
        sizes["Rotation"] = sizes["Main dataset"];
        sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        sizes["Z stats"] = Stats::size({height, width});
    }
    
    hsize_t total(0);
    
    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;
    
    for (auto& kv : sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
        total += kv.second;
    }
    
    std::string note = "";
    if (depth > 1) {
        total -= std::min(sizes["Mipmaps"], sizes["Rotation"]);
        note = " (Rotated dataset and mipmaps are not allocated at the same time.)";
    }

    std::cout << "TOTAL:\t" << total * 1e-9 << "GB" << note << std::endl;
}

void FastConverter::copyAndCalculate() {
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
    
    hsize_t& numBinsXY(numBins);
    hsize_t& numBinsXYZ(numBins);
    std::string timerLabelXYRotation = depth > 1 ? "XY statistics and rotation" : "XY statistics";

    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        DEBUG(std::cout << "Processing Stokes " << currentStokes << "..." << std::endl;);

        // Read data into memory space
        TIMER(timer.start("Read"););
        DEBUG(std::cout << "+ Reading main dataset..." << std::flush;);
        readFitsData(inputFilePtr, 0, currentStokes, cubeSize, standardCube);
        
        // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
        if (depth > 1) {
            TIMER(timer.start("Allocate"););
            rotatedCube = new float[cubeSize];
        }
        
        DEBUG(std::cout << " " << timerLabelXYRotation <<  "..." << std::flush;);
        TIMER(timer.start(timerLabelXYRotation););

        // First loop calculates stats for each XY slice and rotates the dataset
#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
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
            
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    auto sourceIndex = k + width * j + (height * width) * i;
                    auto destIndex = i + depth * j + (height * depth) * k;
                    auto& val = standardCube[sourceIndex];
                    
                    if (depth > 1) {
                        rotatedCube[destIndex] = val;
                    }
                    
                    if (std::isfinite(val)) {
                        minmax(val);
                        sum += val;
                        sumSq += val * val;
                    } else {
                        nanCount += 1;
                    }
                }
            }

            auto& indexXY = i;
            
            statsXY.nanCounts[indexXY] = nanCount;
            statsXY.sums[indexXY] = sum;
            statsXY.sumsSq[indexXY] = sumSq;
            
            if ((hsize_t)nanCount != (height * width)) {
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
            // Consolidate XY stats into XYZ stats
            DEBUG(std::cout << " XYZ statistics..." << std::flush;);
            TIMER(timer.start("XYZ and Z statistics"););
            
            double xyzSum = 0;
            double xyzSumSq = 0;
            int64_t xyzNanCount = 0;
            xyzMin = statsXY.minVals[0];
            xyzMax = statsXY.maxVals[0];

            for (hsize_t i = 0; i < depth; i++) {
                auto& indexXY = i;
                if (std::isfinite(statsXY.maxVals[indexXY])) {
                    xyzSum += statsXY.sums[indexXY];
                    xyzSumSq += statsXY.sumsSq[indexXY];
                    xyzMin = fmin(xyzMin, statsXY.minVals[indexXY]);
                    xyzMax = fmax(xyzMax, statsXY.maxVals[indexXY]);
                }
                xyzNanCount += statsXY.nanCounts[indexXY];
            }

            statsXYZ.nanCounts[0] = xyzNanCount;
            statsXYZ.sums[0] = xyzSum;
            statsXYZ.sumsSq[0] = xyzSumSq;
            
            if ((hsize_t)xyzNanCount != depth * height * width) {
                statsXYZ.minVals[0] = xyzMin;
                statsXYZ.maxVals[0] = xyzMax;
            } else {
                statsXYZ.minVals[0] = NAN;
                statsXYZ.maxVals[0] = NAN;
            }

            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
            
            DEBUG(std::cout << " Z statistics... " << std::flush;);
            
#pragma omp parallel for
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    float minVal = std::numeric_limits<float>::max();
                    float maxVal = -std::numeric_limits<float>::max();
                    double sum = 0;
                    double sumSq = 0;
                    int64_t nanCount = 0;
                    
                    for (hsize_t i = 0; i < depth; i++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto& val = standardCube[sourceIndex];

                        if (std::isfinite(val)) {
                            // Not replacing this with if/else; too much risk of encountering an ascending / descending sequence.
                            minVal = fmin(minVal, val);
                            maxVal = fmax(maxVal, val);
                            sum += val;
                            sumSq += val * val;
                        } else {
                            nanCount += 1;
                        }
                    }
                    
                    auto indexZ = k + j * width;
                    
                    statsZ.nanCounts[indexZ] = nanCount;
                    statsZ.sums[indexZ] = sum;
                    statsZ.sumsSq[indexZ] = sumSq;
                    
                    if ((hsize_t)nanCount != depth) {
                        statsZ.minVals[indexZ] = minVal;
                        statsZ.maxVals[indexZ] = maxVal;
                    } else {
                        statsZ.minVals[indexZ] = NAN;
                        statsZ.maxVals[indexZ] = NAN;
                    }
                }
            }
        
        }

        // Third loop handles histograms
        
        DEBUG(std::cout << " Histograms..." << std::flush;);
        TIMER(timer.start("Histograms"););
        
        double cubeMin;
        double cubeMax;
        double cubeRange;
                    
        if (depth > 1) {
            cubeMin = xyzMin;
            cubeMax = xyzMax;
            cubeRange = cubeMax - cubeMin;
        }

#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            auto& indexXY = i;
            double sliceMin = statsXY.minVals[indexXY];
            double sliceMax = statsXY.maxVals[indexXY];
            double range = sliceMax - sliceMin;
            
            std::function<void(float)> channelHistogramFunc;
            std::function<void(float)> cubeHistogramFunc;
            
            auto doChannelHistogram = [&] (float val) {
                // XY Histogram
                int binIndex = std::min(numBinsXY - 1, (hsize_t)(numBinsXY * (val - sliceMin) / range));
                statsXY.histograms[i * numBinsXY + binIndex]++;
            };
            
            auto doCubeHistogram = [&] (float val) {
                // XYZ Partial histogram
                int binIndexXYZ = std::min(numBinsXYZ - 1, (hsize_t)(numBinsXYZ * (val - cubeMin) / cubeRange));
                statsXYZ.partialHistograms[i * numBinsXYZ + binIndexXYZ]++;
            };
            
            auto doNothing = [&] (float val) {
                UNUSED(val);
            };
            
            channelHistogramFunc = doChannelHistogram;
            cubeHistogramFunc = doCubeHistogram;
            bool chanHist(true);
            bool cubeHist(true);
            
            if (!std::isfinite(sliceMin) || !std::isfinite(sliceMax) || range == 0) {
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
            for (hsize_t i = 0; i < depth; i++) {
                for (hsize_t j = 0; j < numBinsXYZ; j++) {
                    statsXYZ.histograms[j] +=
                        statsXYZ.partialHistograms[i * numBinsXYZ + j];
                }
            }
        }

        DEBUG(std::cout << " Writing main and rotated datasets... " << std::flush;);
        TIMER(timer.start("Write"););
                    
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

        // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
        if (depth > 1) {
            DEBUG(std::cout << " Freeing memory from rotated dataset..." << std::flush;);
            TIMER(timer.start("Free"););
            
            delete[] rotatedCube;
        }
        
        // Fourth loop handles mipmaps
        
        // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
        DEBUG(std::cout << " Mipmaps..." << std::endl;);
        TIMER(timer.start("Mipmaps"););
        
#pragma omp parallel for
        for (hsize_t c = 0; c < depth; c++) {
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
        
        // Final mipmap calculation
        mipMaps.calculate();
        
        TIMER(timer.start("Write"););
        
        // Write the mipmaps
        mipMaps.write(currentStokes, 0);
        
        // Write the statistics                
        statsXY.write({1, depth}, {currentStokes, 0});
        
        if (depth > 1) {
            statsXYZ.write({1}, {currentStokes});
            statsZ.write({1, height, width}, {currentStokes, 0, 0});
        }
        
        // Clear the stats before the next Stokes
        TIMER(timer.start(timerLabelXYRotation););
        
        statsXY.resetBuffers();
        
        TIMER(timer.start("XYZ and Z statistics"););
        
        if (depth > 1) {
            statsXYZ.resetBuffers();
            statsZ.resetBuffers();
        }
                
        // Clear the mipmaps before the next Stokes
        TIMER(timer.start("Mipmaps"););
        mipMaps.resetBuffers();
        
    } // end of Stokes loop
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
}
