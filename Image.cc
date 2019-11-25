#include "Image.h"

Image::Image(std::string inputFileName, std::string outputFileName, bool slow) :
    status(0),
    strType(H5::PredType::C_S1, 256),
    boolType(H5::PredType::NATIVE_HBOOL), 
    doubleType(H5::PredType::NATIVE_DOUBLE),
    floatType(H5::PredType::NATIVE_FLOAT),
    intType(H5::PredType::NATIVE_INT64)
{        
    fits_open_file(&inputFilePtr, inputFileName.c_str(), READONLY, &status);
    
    if (status != 0) {
        throw "Could not open FITS file";
    }
    
    int bitpix;
    fits_get_img_type(inputFilePtr, &bitpix, &status);
    
    if (status != 0) {
        throw "Could not read image type";
    }

    if (bitpix != -32) {
        throw "Currently only supports FP32 files";
    }
    
    fits_get_img_dim(inputFilePtr, &N, &status);
    
    if (status != 0) {
        throw "Could not read image dimensions";
    }

    if (N < 2 || N > 4) {
        throw "Currently only supports 2D, 3D and 4D cubes";
    }
    
    long dims[4];
    fits_get_img_size(inputFilePtr, 4, dims, &status);
    
    if (status != 0) {
        throw "Could not read image size";
    }
        
    stokes = N == 4 ? dims[3] : 1;
    depth = N >= 3 ? dims[2] : 1;
    height = dims[1];
    width = dims[0];
    
    swizzledName = N == 3 ? "ZYX" : "ZYXW";
    
    MipMap::initialise(mipMaps, N, width, height, slow ? 1 : depth);
    
    this->slow = slow;
    timer = Timer(stokes * depth * height * width, slow);
    
    // Customise types
    doubleType.setOrder(H5T_ORDER_LE);
    floatType.setOrder(H5T_ORDER_LE);
    intType.setOrder(H5T_ORDER_LE);
    
    // Dimensions of various datasets
    this->dims = Dims::makeDims(N, dims);
    
    numBinsXY = this->dims.statsXY.numBins;        
    if (depth > 1) {
        numBinsXYZ = this->dims.statsXYZ.numBins;
    }
    
    // Prepare output file
    this->outputFileName = outputFileName;
    tempOutputFileName = outputFileName + ".tmp";        
}

Image::~Image() {
    outputFile.close();
}

void Image::createOutputFile() {
    outputFile = H5::H5File(tempOutputFileName, H5F_ACC_TRUNC);
    outputGroup = outputFile.createGroup("0");
    
    H5::DSetCreatPropList standardCreatePlist;
    if (dims.useChunks()) {
        standardCreatePlist.setChunk(N, dims.tileDims.data());
    }
    auto standardDataSpace = H5::DataSpace(N, dims.standard.data());
    standardDataSet = outputGroup.createDataSet("DATA", floatType, standardDataSpace, standardCreatePlist);

    if (depth > 1) {
        auto swizzledGroup = outputGroup.createGroup("SwizzledData");
        auto swizzledDataSpace = H5::DataSpace(N, dims.swizzled.data());
        swizzledDataSet = swizzledGroup.createDataSet(swizzledName, floatType, swizzledDataSpace);
    }
    
    if (mipMaps.size()) {
        // I don't know if this naming convention still makes sense, but I'm replicating the schema for now
        auto mipMapGroup = outputGroup.createGroup("MipMaps").createGroup("DATA");
        
        for (auto& mipMap : mipMaps) {
            mipMap.createDataset(mipMapGroup, floatType, dims);
        }
    }
}

void Image::copyHeaders() {
    H5::DataSpace attributeDataSpace(H5S_SCALAR);
    
    H5::Attribute attribute = outputGroup.createAttribute("SCHEMA_VERSION", strType, attributeDataSpace);
    attribute.write(strType, SCHEMA_VERSION);
    attribute = outputGroup.createAttribute("HDF5_CONVERTER", strType, attributeDataSpace);
    attribute.write(strType, HDF5_CONVERTER);
    attribute = outputGroup.createAttribute("HDF5_CONVERTER_VERSION", strType, attributeDataSpace);
    attribute.write(strType, HDF5_CONVERTER_VERSION);

    int numHeaders;
    fits_get_hdrspace(inputFilePtr, &numHeaders, NULL, &status);
    
    if (status != 0) {
        throw "Could not read image header";
    }
    
    char keyTmp[255];
    char valueTmp[255];
    
    for (auto i = 0; i < numHeaders; i++) {
        fits_read_keyn(inputFilePtr, i, keyTmp, valueTmp, NULL, &status);
    
        if (status != 0) {
            throw "Could not read attribute from header";
        }
        std::string attributeName(keyTmp);
        std::string attributeValue(valueTmp);
        
        if (attributeName.empty() || attributeName.find("COMMENT") == 0 || attributeName.find("HISTORY") == 0) {
            // TODO we should actually do something about these
        } else {
            if (outputGroup.attrExists(attributeName)) {
                std::cout << "Warning: Skipping duplicate attribute '" << attributeName << "'" << std::endl;
            } else {
                bool parsingFailure(false);
                
                if (attributeValue.length() >= 2 && attributeValue.find('\'') == 0 &&
                    attributeValue.find_last_of('\'') == attributeValue.length() - 1) {
                    // STRING
                    int strLen;
                    char strValueTmp[255];
                    fits_read_string_key(inputFilePtr, attributeName.c_str(), 1, 255, strValueTmp, &strLen, NULL, &status);
    
                    if (status != 0) {
                        throw "Could not read string attribute";
                    }
                    
                    std::string attributeValueStr(strValueTmp);

                    attribute = outputGroup.createAttribute(attributeName, strType, attributeDataSpace);
                    attribute.write(strType, attributeValueStr);
                } else if (attributeValue == "T" || attributeValue == "F") {
                    // BOOLEAN
                    bool attributeValueBool = (attributeValue == "T");
                    attribute = outputGroup.createAttribute(attributeName, boolType, attributeDataSpace);
                    attribute.write(boolType, &attributeValueBool);
                } else if (attributeValue.find('.') != std::string::npos) {
                    // TRY TO PARSE AS DOUBLE
                    try {
                        double attributeValueDouble = std::stod(attributeValue);
                        attribute = outputGroup.createAttribute(attributeName, doubleType, attributeDataSpace);
                        attribute.write(doubleType, &attributeValueDouble);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: Could not parse attribute '" << attributeName << "' as a float." << std::endl;
                        parsingFailure = true;
                    }
                } else {
                    // TRY TO PARSE AS INTEGER
                    try {
                        int64_t attributeValueInt = std::stoi(attributeValue);
                        attribute = outputGroup.createAttribute(attributeName, intType, attributeDataSpace);
                        attribute.write(intType, &attributeValueInt);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: Could not parse attribute '" << attributeName << "' as an integer." << std::endl;
                        parsingFailure = true;
                    }
                }
                
                if (parsingFailure) {
                    // FALL BACK TO STRING
                    attribute = outputGroup.createAttribute(attributeName, strType, attributeDataSpace);
                    attribute.write(strType, attributeValue);
                }
            }
        }
    }
}

void Image::slowSwizzle() {
    hsize_t sliceSize;
    
    if (N == 3) {
        sliceSize = depth * TILE_SIZE * TILE_SIZE;
    } else if (N == 4) {
        sliceSize = stokes * depth * TILE_SIZE * TILE_SIZE;
    }
    
    std::cout << "Allocating " << sliceSize * 4 * 2 * 1e-9 << " GB of memory... " << std::endl;
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
    
    delete[] standardSlice;
    delete[] rotatedSlice;
}

void Image::allocate(hsize_t cubeSize) {
    std::cout << "Allocating " << cubeSize * 4 * 2 * 1e-9 << " GB of memory... " << std::endl;
    timer.alloc.start();

    standardCube = new float[cubeSize];
    
    statsXY = Stats(dims.statsXY);
    
    if (depth > 1) {
        statsZ = Stats(dims.statsZ);
        statsXYZ = Stats(dims.statsXYZ);
    }

    timer.alloc.stop();
}

void Image::allocateSwizzled(hsize_t rotatedSize) {
    if (depth > 1) {
        timer.alloc.start();
        rotatedCube = new float[rotatedSize];
        timer.alloc.stop();
    }
}

void Image::freeSwizzled() {
    if (depth > 1) {
        delete[] rotatedCube;
    }
}

void Image::readFits(long* fpixel, int cubeSize) {
    timer.read.start();
    fits_read_pix(inputFilePtr, TFLOAT, fpixel, cubeSize, NULL, standardCube, NULL, &status);
    timer.read.stop();
    
    if (status != 0) {
        throw "Could not read image data";
    }
}

void Image::fastCopy() {
    auto cubeSize = depth * height * width;
    allocate(cubeSize);

    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        // Read data into memory space
        
        long fpixel[] = {1, 1, 1, currentStokes + 1};
        std::cout << "Reading Stokes " << currentStokes << " dataset... " << std::endl;
        readFits(fpixel, cubeSize);
        
        // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
        allocateSwizzled(cubeSize);

        std::cout << "Processing Stokes " << currentStokes << " dataset..." << std::endl;
        timer.process1.start();
        
        std::cout << " * XY statistics" << (depth > 1 ? " and fast swizzling" : "") <<  "... " << std::endl;

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
        }
        
        timer.process1.stop();

        if (depth > 1) {
            std::cout << " * Z statistics... " << std::endl;
            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
            
            timer.process2.start();
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
            
            timer.process2.stop();
        }
        
        std::cout << " * Histograms... " << std::endl;

        // Third loop handles histograms
        
        timer.process3.start();
        
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

        timer.process3.stop();

        std::cout << "Writing Stokes " << currentStokes << " dataset... " << std::endl;
        
        timer.write.start();
                    
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

        timer.write.stop();
        
        // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
        
        freeSwizzled();
        
        // Fourth loop handles mipmaps
        
        // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
        
        std::cout << " * MipMaps... " << std::endl;

        timer.process4.start();
        
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
        
        timer.process4.stop();
        timer.write.start();
        
        // Write the mipmaps
        
        for (auto& mipMap : mipMaps) {
            // Start at current Stokes and channel 0
            mipMap.write(currentStokes, 0);
        }
        
        timer.write.stop();
        
        // Clear the mipmaps before the next Stokes
        
        timer.process4.start();
        
        for (auto& mipMap : mipMaps) {
            mipMap.reset();
        }
        
        timer.process4.stop();
    } // end of Stokes loop
}

void Image::slowCopy() {
    // Allocate one channel at a time, and no swizzled data
    auto cubeSize = height * width;
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
        slowSwizzle();
    }
}

void Image::convert() {
    createOutputFile();
    copyHeaders();
    
    if (slow) {
        // Slow, memory-saving algorithm
        slowCopy();
    } else {
        // Fast algorithm
        fastCopy();
    }
    
    // Write statistics
    timer.write.start();
    auto statsGroup = outputGroup.createGroup("Statistics");
    
    auto statsXYGroup = statsGroup.createGroup("XY"); 
    statsXY.write(statsXYGroup, floatType, intType);

    if (depth > 1) {
        auto statsXYZGroup = statsGroup.createGroup("XYZ"); 
        auto statsZGroup = statsGroup.createGroup("Z"); 
        statsXYZ.write(statsXYZGroup, floatType, intType);
        statsZ.write(statsZGroup, floatType, intType);
    }
    timer.write.stop();
    
    // Free memory
    delete[] standardCube;
    
    // Rotated cube is freed elsewhere
            
    timer.print();
    
    // Rename from temp file
    rename(tempOutputFileName.c_str(), outputFileName.c_str());
}
