/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

Converter::Converter(std::string inputFileName, std::string outputFileName, bool progress) : timer(), progress(progress) {
    TIMER(timer.start("Setup"););
    
    openFitsFile(&inputFilePtr, inputFileName);
    
    long dims[4];
    
    getFitsDims(inputFilePtr, N, dims);
        
    stokes = N == 4 ? dims[3] : 1;
    depth = N >= 3 ? dims[2] : 1;
    height = dims[1];
    width = dims[0];
    
    swizzledName = N == 3 ? "ZYX" : "ZYXW";
    
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
    mipMaps = MipMaps(standardDims, tileDims);
    
    // Prepare output file
    this->outputFileName = outputFileName;
    tempOutputFileName = outputFileName + ".tmp";        
}

Converter::~Converter() {
    // TODO this is probably unnecessary; the file object destructor should close the file properly.
    outputFile.close();
}

std::unique_ptr<Converter> Converter::getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool progress) {
    if (slow) {
        return std::unique_ptr<Converter>(new SlowConverter(inputFileName, outputFileName, progress));
    } else {
        return std::unique_ptr<Converter>(new FastConverter(inputFileName, outputFileName, progress));
    }
}

void Converter::copyAndCalculate() {
    // implemented in subclasses
}

void Converter::reportMemoryUsage() {
    // implemented in subclasses
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
    
    // IMPORTANT: This is 1-indexed!
    for (int i = 1; i <= numAttributes; i++) {        
        std::string attributeName;
        std::string attributeValue;
        readFitsAttribute(inputFilePtr, i, attributeName, attributeValue);
        
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
                    writeHdf5Attribute(outputGroup, attributeName, attributeValueStr);
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
