/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

Converter::Converter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) : timer(), progress(progress), zMips(zMips), swapStokesFreqAxis(false) {
    TIMER(timer.start("Setup"););
    
    openFitsFile(&inputFilePtr, inputFileName);
    
    long dims[4];
    
    getFitsDims(inputFilePtr, N, dims);
    
    // check relevant FITS keywords if the axis order is STOKES,FREQ -> swap is required:
    checkIfSwapAxisRequired();
        
    stokes = N == 4 ? dims[3] : 1;
    depth = N >= 3 ? dims[2] : 1;
    if ( swapStokesFreqAxis ) {
       stokes = N >= 4 ? dims[2] : 1;
       depth = N == 4 ? dims[3] : 1;
    }
    height = dims[1];
    width = dims[0];
    
    swizzledName = N == 3 ? "ZYX" : "ZYXW";
    
    DEBUG(std::cout << "Stokes = " << stokes << ", depth = " << depth << ", image dimensions " << height << " x " << width << " , swizzledName = " << swizzledName.c_str(););
    
    standardDims = trimAxes({stokes, depth, height, width}, N);
    tileDims = trimAxes({1, 1, TILE_SIZE, TILE_SIZE}, N);
    
    numBins = int(std::max(std::sqrt(width * height), 2.0));
    
    // STATS OBJECTS

    auto statsXYDims = trimAxes({stokes, depth}, N - 2);
    statsXY = Stats(statsXYDims, numBins);
    
    if (depth > 1) {
        swizzledDims = trimAxes({stokes, width, height, depth}, N);
        statsZ = Stats(trimAxes({stokes, height, width}, N - 1));
        auto statsXYZDims = trimAxes({stokes}, N - 3);
        statsXYZ = Stats(statsXYZDims, numBins);
    }
    
    // MIPMAPS
    mipMaps = MipMaps(standardDims, tileDims, zMips);
    
    // Prepare output file
    this->outputFileName = outputFileName;
    tempOutputFileName = outputFileName + ".tmp";        
}

Converter::~Converter() {
    // TODO this is probably unnecessary; the file object destructor should close the file properly.
    outputFile.close();
    closeFitsFile(inputFilePtr);
}

std::unique_ptr<Converter> Converter::getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool progress, bool zMips) {
    if (slow) {
        return std::unique_ptr<Converter>(new SlowConverter(inputFileName, outputFileName, progress, zMips));
    } else {
        return std::unique_ptr<Converter>(new FastConverter(inputFileName, outputFileName, progress, zMips));
    }
}

void Converter::reportMemoryUsage() {
    MemoryUsage m = calculateMemoryUsage();

    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;
    
    for (auto& kv : m.sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
    }

    std::cout << "TOTAL:\t" << m.total * 1e-9 << "GB" << m.note << std::endl;
}

bool Converter::checkIfSwapAxisRequired() {
    int numAttributes;
    readFitsHeader(inputFilePtr, numAttributes);

    bool bFreqAxisFound = false;

    swapStokesFreqAxis = false;
    // Check if both STOKES and FREQ axis are present and the order is STOKES,FREQ:
    for (int i = 1; i <= numAttributes; i++) {
       std::string attributeName;
       std::string attributeValue;
       readFitsAttribute(inputFilePtr, i, attributeName, attributeValue);
       DEBUG(std::cout << "|" << attributeName.c_str() << "| = |" << attributeValue.c_str() << "|" << std::endl;);

       if (attributeName == "CTYPE3" && attributeValue.find("STOKES") != std::string::npos) {
          std::cout << "INFO : detected 3rd axis CTYPE3 = STOKES" << std::endl;
          swapStokesFreqAxis = true;
       }
       if (attributeName == "CTYPE4" && attributeValue.find("FREQ") != std::string::npos) {
          std::cout << "INFO : detected 4th axis CTYPE4 = FREQ" << std::endl;
          bFreqAxisFound = true;
       }
    }
    if (swapStokesFreqAxis) {
       if (!bFreqAxisFound) {
          swapStokesFreqAxis = false; // If Stokes is 3rd axis, but 4th axis is not frequency -> no swap
       }
    }
    std::cout << "INFO : swapStokesFreqAxis = " << swapStokesFreqAxis << std::endl;
    
    return swapStokesFreqAxis;
}

void Converter::convert() {
    // CREATE OUTPUT FILE
    
    // TODO dataset variables should be local and passed into the copy function?
    
    outputFile = H5::H5File(tempOutputFileName, H5F_ACC_TRUNC);
    outputGroup = outputFile.createGroup("0");
    
    std::vector<hsize_t> chunkDims;
    if (useChunks(standardDims)) {
        chunkDims = tileDims;
    }
    
    H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
    floatType.setOrder(H5T_ORDER_LE);
    createHdf5Dataset(standardDataSet, outputGroup, "DATA", floatType, standardDims, chunkDims);
    
    statsXY.createDatasets(outputGroup, "XY");

    if (depth > 1) {
        statsXYZ.createDatasets(outputGroup, "XYZ");
        statsZ.createDatasets(outputGroup, "Z");
        
        auto swizzledGroup = outputGroup.createGroup("SwizzledData");
        // We use this name in papers because it sounds more serious. :)
        outputGroup.link(H5L_TYPE_HARD, "SwizzledData", "PermutedData");
        createHdf5Dataset(swizzledDataSet, swizzledGroup, swizzledName, floatType, swizzledDims);
    }
    
    mipMaps.createDatasets(outputGroup);
    
    // COPY HEADERS
    
    TIMER(timer.start("Headers"););
    
    writeHdf5Attribute(outputGroup, "SCHEMA_VERSION", std::string(SCHEMA_VERSION));
    writeHdf5Attribute(outputGroup, "HDF5_CONVERTER", std::string(HDF5_CONVERTER));
    writeHdf5Attribute(outputGroup, "HDF5_CONVERTER_VERSION", std::string(HDF5_CONVERTER_VERSION));

    int numAttributes;
    readFitsHeader(inputFilePtr, numAttributes);
    
    std::vector<std::string> wcsStokesKeywords = {"CTYPE3","CRVAL3","CDELT3","CRPIX3","CUNIT3"};
    std::vector<std::string> wcsFreqKeywords = {"CTYPE4","CRVAL4","CDELT4","CRPIX4","CUNIT4"};
    
    // IMPORTANT: This is 1-indexed!
    for (int i = 1; i <= numAttributes; i++) {
        bool bSwappingStokesFreqAxisNow = false;
        std::string attributeName;
        std::string attributeValue;
        readFitsAttribute(inputFilePtr, i, attributeName, attributeValue);
 
        if (swapStokesFreqAxis) { // check if swap of STOKES <-> FREQ axis is required:
           // If Stokes is 3rd axis and Freq 4th axis -> SWAP Stokes and Frequency axis in FITS header
           if (std::find(wcsStokesKeywords.begin(), wcsStokesKeywords.end(), attributeName) != wcsStokesKeywords.end()) {
              if(attributeName.back() == '3') {
                 // if swap of STOKES and FREQ axis is required change 3 -> 4:
                 std::cout << "INFO : swapping attribute " << attributeName << " (3 -> 4) " << std::endl;
                 attributeName.back() = '4';
                 bSwappingStokesFreqAxisNow = true;
              }
           }else{
              if ( std::find(wcsFreqKeywords.begin(), wcsFreqKeywords.end(), attributeName) != wcsFreqKeywords.end() ){
                 if(attributeName.back() == '4') {
                    // if swap of STOKES and FREQ axis is required change 4 -> 3:
                    std::cout << "INFO : swapping attribute " << attributeName << " (4 -> 3) " << std::endl;
                    attributeName.back() = '3';
                    bSwappingStokesFreqAxisNow = true;
                 }
              }
           }
        }
        
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
                    std::string attributeValueStr;
                    readFitsStringAttribute(inputFilePtr, attributeName, attributeValueStr);
                    // this may not be required 
                    if( bSwappingStokesFreqAxisNow ) {
                       // if swapping axis now, we have to use attributeValue as the attributes's name has been swapped 
                       // for example CTYPE3 -> CTYPE4 which means value of CTYPE4 would be read (FREQ) and no swap would happen!
                       attributeValueStr = attributeValue;
                    }
                    writeHdf5Attribute(outputGroup, attributeName, attributeValueStr);
                    DEBUG(std::cout << "Written attribute " << attributeName.c_str() << " = " << attributeValueStr.c_str() << std::endl;);
                } else if (attributeValue == "T" || attributeValue == "F") {
                    // BOOLEAN
                    bool attributeValueBool = (attributeValue == "T");
                    writeHdf5Attribute(outputGroup, attributeName, attributeValueBool);
                } else if (attributeValue.find('.') != std::string::npos) {
                    // TRY TO PARSE AS DOUBLE
                    try {
                        double attributeValueDouble = std::stod(attributeValue);
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueDouble);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as a float." << std::endl;
                        parsingFailure = true;
                    } catch (const std::out_of_range& e) {
                        // Special handling for subnormal numbers
                        long double attributeValueLongDouble = std::stold(attributeValue);
                        double attributeValueDouble = (double) attributeValueLongDouble;
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueDouble);
                        
                        std::ostringstream ostream;
                        ostream.precision(13);
                        ostream << attributeValueDouble;
                        std::string original(attributeValue);
                        std::string round_trip(ostream.str());
                        transform(original.begin(), original.end(), original.begin(), ::toupper);
                        transform(round_trip.begin(), round_trip.end(), round_trip.begin(), ::toupper);
                                                
                        if (original != round_trip) {
                            std::cout << "Warning: the value of attribute  '" << attributeName << "' is not representable as a normalised double precision floating point number. Some precision has been lost.\nOriginal string representation:\n'" << original << "'\nFinal string representation:\n'" << round_trip << "'" << std::endl;
                        }
                    }
                } else {
                    // TRY TO PARSE AS INTEGER
                    try {
                        int64_t attributeValueInt = std::stoi(attributeValue);
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueInt);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as an integer." << std::endl;
                        parsingFailure = true;
                    }
                }
                
                if (parsingFailure) {
                    // FALL BACK TO STRING
                    writeHdf5Attribute(outputGroup, attributeName, attributeValue);
                }
            }
        }
    }
    
    // MAIN CONVERSION AND CALCULATION FUNCTION

    copyAndCalculate();
            
    TIMER(timer.print(product(standardDims)););
    
    // Rename from temp file
    rename(tempOutputFileName.c_str(), outputFileName.c_str());
}
