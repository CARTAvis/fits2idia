#include "Converter.h"

Converter::Converter(std::string inputFileName, std::string outputFileName) :
    status(0),
    timer(),
    strType(H5::PredType::C_S1, 256),
    boolType(H5::PredType::NATIVE_HBOOL), 
    doubleType(H5::PredType::NATIVE_DOUBLE),
    floatType(H5::PredType::NATIVE_FLOAT),
    intType(H5::PredType::NATIVE_INT64)
{
    T(timer.start("Setup"););
    
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

Converter::~Converter() {
    // TODO this is probably unnecessary; the file object destructor should close the file properly.
    outputFile.close();
}

std::unique_ptr<Converter> Converter::getConverter(std::string inputFileName, std::string outputFileName, bool slow) {
    if (slow) {
        return std::unique_ptr<Converter>(new SlowConverter(inputFileName, outputFileName));
    } else {
        return std::unique_ptr<Converter>(new FastConverter(inputFileName, outputFileName));
    }
}

void Converter::readFits(hsize_t channel, unsigned int stokes, hsize_t size, float* destination) {
    long fpixel[] = {1, 1, (long)channel + 1, stokes + 1};
    fits_read_pix(inputFilePtr, TFLOAT, fpixel, size, NULL, destination, NULL, &status);
    
    if (status != 0) {
        throw "Could not read image data";
    }
}

void Converter::copyAndCalculate() {
    // implemented in subclasses
}

void Converter::convert() {
    // CREATE OUTPUT FILE
    
    outputFile = H5::H5File(tempOutputFileName, H5F_ACC_TRUNC);
    outputGroup = outputFile.createGroup("0");
    
    H5::DSetCreatPropList standardCreatePlist;
    if (dims.useChunks()) {
        standardCreatePlist.setChunk(N, dims.tileDims.data());
    }
    auto standardDataSpace = H5::DataSpace(N, dims.standard.data());
    standardDataSet = outputGroup.createDataSet("DATA", floatType, standardDataSpace, standardCreatePlist);
    
    auto statsGroup = outputGroup.createGroup("Statistics");    
    auto statsXYGroup = statsGroup.createGroup("XY");

    if (depth > 1) {
        auto statsXYZGroup = statsGroup.createGroup("XYZ"); 
        auto statsZGroup = statsGroup.createGroup("Z");
        
        auto swizzledGroup = outputGroup.createGroup("SwizzledData");
        // We use this name in papers because it sounds more serious. :)
        outputGroup.link(H5L_TYPE_HARD, "SwizzledData", "PermutedData");
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
    
    // COPY HEADERS
    
    T(timer.start("Headers"););
    
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
    
    // This is 1-indexed!
    for (auto i = 1; i <= numHeaders; i++) {
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
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as a float." << std::endl;
                        parsingFailure = true;
                    } catch (const std::out_of_range& e) {
                        long double attributeValueLongDouble = std::stold(attributeValue);
                        double attributeValueDouble = (double) attributeValueLongDouble;
                        attribute = outputGroup.createAttribute(attributeName, doubleType, attributeDataSpace);
                        attribute.write(doubleType, &attributeValueDouble);
                        
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
                        attribute = outputGroup.createAttribute(attributeName, intType, attributeDataSpace);
                        attribute.write(intType, &attributeValueInt);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as an integer." << std::endl;
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
    
    // MAIN CONVERSION AND CALCULATION FUNCTION

    copyAndCalculate();
    
    // WRITE STATISTICS
    
    // TODO only store stats for one stokes at a time; stats don't span multiple stokes.
    
    T(timer.start("Write"););

    statsXY.write(statsXYGroup, floatType, intType);
    if (depth > 1) {
        auto statsXYZGroup = statsGroup.openGroup("XYZ"); 
        auto statsZGroup = statsGroup.openGroup("Z");
        
        statsXYZ.write(statsXYZGroup, floatType, intType);
        statsZ.write(statsZGroup, floatType, intType);
    }
    
    // Free memory
    T(timer.start("Free"););
    D(std::cout << "Freeing memory from main dataset... " << std::endl;);
    delete[] standardCube;
    
    // Rotated cube is freed elsewhere
            
    T(timer.print(stokes * depth * height * width););
    
    // Rename from temp file
    rename(tempOutputFileName.c_str(), outputFileName.c_str());
}
