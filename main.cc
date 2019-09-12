#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <getopt.h>

#include <H5Cpp.h>
#include <fitsio.h>

using namespace H5;
using namespace std;

#define SCHEMA_VERSION "0.2"
#define HDF5_CONVERTER "hdf_convert"
#define HDF5_CONVERTER_VERSION "0.1.8"

struct Types {
    Types() :
        strType(PredType::C_S1, 256),
        boolType(PredType::NATIVE_HBOOL), 
        doubleType(PredType::NATIVE_DOUBLE),
        floatType(PredType::NATIVE_FLOAT),
        intType(PredType::NATIVE_INT64)
    {
        doubleType.setOrder(H5T_ORDER_LE);
        floatType.setOrder(H5T_ORDER_LE);
        intType.setOrder(H5T_ORDER_LE);
    }
    
    StrType strType;
    IntType boolType;
    FloatType doubleType;
    FloatType floatType;
    IntType intType;
};

bool getOptions(int argc, char** argv, string& inputFileName, string& outputFileName, bool& slow) {
    extern int optind;
    extern char *optarg;
    
    int opt;
    bool err(false);
    string usage = "Usage: hdf_convert [-o output_filename] [-s] input_filename\n\nConvert a FITS file to an HDF5 file with the IDIA schema\n\nOptions:\n\n-o\tOutput filename\n-s\tUse slower but less memory-intensive method (enable if memory allocation fails)";
    
    while ((opt = getopt(argc, argv, ":o:s")) != -1) {
        switch (opt) {
            case 'o':
                // TODO: output filename
                outputFileName.assign(optarg);
                break;
            case 's':
                // use slower but less memory-intensive method
                slow = true;
                break;
            case ':':
                err = true;
                cerr << "Missing argument for option " << opt << "." << endl;
                break;
            case '?':
                err = true;
                cerr << "Unknown option " << opt << "." << endl;
                break;
        }
    }
    
    if (optind >= argc) {
        err = true;
        cerr << "Missing input filename parameter." << endl;
    } else {
        inputFileName.assign(argv[optind]);
        optind++;
    }
    
    if (argc > optind) {
        err = true;
        cerr << "Unexpected additional parameters." << endl;
    }
        
    if (err) {
        cerr << usage << endl;
        return false;
    }
    
    if (outputFileName.empty()) {
        auto fitsIndex = inputFileName.find_last_of(".fits");
        if (fitsIndex != string::npos) {
            outputFileName = inputFileName.substr(0, fitsIndex - 4);
            outputFileName += ".hdf5";
        } else {
            outputFileName = inputFileName + ".hdf5";
        }
    }
    
    return true;
}

bool openFitsFile(string inputFileName, fitsfile*& inputFilePtr) {
    int status = 0;
    int bitpix;
    
    fits_open_file(&inputFilePtr, inputFileName.c_str(), READONLY, &status);
    
    if (status != 0) {
        cerr << "error opening FITS file" << endl;
        return false;
    }
    
    fits_get_img_type(inputFilePtr, &bitpix, &status);

    if (bitpix != -32) {
        cerr << "Currently only supports FP32 files" << endl;
        return false;
    }

    return true;
}

bool getDims(fitsfile* inputFilePtr, int& N, long& stokes, long& depth, long& height, long& width) {
    int status = 0;
    
    fits_get_img_dim(inputFilePtr, &N, &status);
    
    if (N < 2 || N > 4) {
        cerr << "Currently only supports 2D, 3D and 4D cubes" << endl;
        return false;
    }
    
    long dims[4];
    fits_get_img_size(inputFilePtr, 4, dims, &status);
        
    stokes = N == 4 ? dims[3] : 1;
    depth = N >= 3 ? dims[2] : 1;
    height = dims[1];
    width = dims[0];
    
    return true;
}

void getDataDims(int N, long stokes, long depth, long height, long width, map<string, vector<hsize_t>>& dataDims, map<string, int>& numBins) {
    numBins["XY"] = int(std::max(sqrt(width * height), 2.0));
    numBins["XYZ"] = numBins["XY"];
    
    dataDims["standard"] = {height, width};
    dataDims["swizzled"] = {width, height};
    dataDims["xyHistogram"] = {numBins["XY"]};
    dataDims["xyStats"] = {};
    dataDims["zStats"] = {height, width};
    dataDims["xyzHistogram"] = {numBins["XYZ"]};
    dataDims["xyzStats"] = {};
    dataDims["tile"] = {512, 512};

    if (N >= 3) {
        for (const auto& d : {"standard", "xyHistogram", "xyStats"}) {
            dataDims[d].insert(dataDims[d].begin(), depth);
        }
        dataDims["swizzled"].push_back(depth);
        dataDims["tile"].insert(dataDims["tile"].begin(), 1);
    }
    
    if (N == 4) {
        for (auto & kv : dataDims) {
            kv.second.insert(kv.second.begin(), stokes);
        }
        dataDims["tile"][0] = 1;
    }
}

void writeHeaders(fitsfile* inputFilePtr, Types types, Group& outputGroup) {
    int status = 0;
    
    DataSpace attributeDataSpace = DataSpace(H5S_SCALAR);
    
    Attribute attribute = outputGroup.createAttribute("SCHEMA_VERSION", types.strType, attributeDataSpace);
    attribute.write(types.strType, SCHEMA_VERSION);
    attribute = outputGroup.createAttribute("HDF5_CONVERTER", types.strType, attributeDataSpace);
    attribute.write(types.strType, HDF5_CONVERTER);
    attribute = outputGroup.createAttribute("HDF5_CONVERTER_VERSION", types.strType, attributeDataSpace);
    attribute.write(types.strType, HDF5_CONVERTER_VERSION);

    int numHeaders;
    fits_get_hdrspace(inputFilePtr, &numHeaders, NULL, &status);
    
    char keyTmp[255];
    char valueTmp[255];
    
    for (auto i = 0; i < numHeaders; i++) {
        fits_read_keyn(inputFilePtr, i, keyTmp, valueTmp, NULL, &status);
        string attributeName(keyTmp);
        string attributeValue(valueTmp);
        
        if (attributeName.empty() || attributeName.find("COMMENT") == 0 || attributeName.find("HISTORY") == 0) {
            // TODO we should actually do something about these
        } else {
            if (outputGroup.attrExists(attributeName)) {
                cout << "Warning: Skipping duplicate attribute '" << attributeName << "'" << endl;
            } else {
                bool parsingFailure(false);
                
                if (attributeValue.length() >= 2 && attributeValue.find('\'') == 0 &&
                    attributeValue.find_last_of('\'') == attributeValue.length() - 1) {
                    // STRING
                    int strLen;
                    char strValueTmp[255];
                    fits_read_string_key(inputFilePtr, attributeName.c_str(), 1, 255, strValueTmp, &strLen, NULL, &status);
                    string attributeValueStr(strValueTmp);

                    attribute = outputGroup.createAttribute(attributeName, types.strType, attributeDataSpace);
                    attribute.write(types.strType, attributeValueStr);
                } else if (attributeValue == "T" || attributeValue == "F") {
                    // BOOLEAN
                    bool attributeValueBool = (attributeValue == "T");
                    attribute = outputGroup.createAttribute(attributeName, types.boolType, attributeDataSpace);
                    attribute.write(types.boolType, &attributeValueBool);
                } else if (attributeValue.find('.') != std::string::npos) {
                    // TRY TO PARSE AS DOUBLE
                    try {
                        double attributeValueDouble = std::stod(attributeValue);
                        attribute = outputGroup.createAttribute(attributeName, types.doubleType, attributeDataSpace);
                        attribute.write(types.doubleType, &attributeValueDouble);
                    } catch (const std::invalid_argument& ia) {
                        cout << "Warning: Could not parse attribute '" << attributeName << "' as a float." << endl;
                        parsingFailure = true;
                    }
                } else {
                    // TRY TO PARSE AS INTEGER
                    try {
                        int64_t attributeValueInt = std::stoi(attributeValue);
                        attribute = outputGroup.createAttribute(attributeName, types.intType, attributeDataSpace);
                        attribute.write(types.intType, &attributeValueInt);
                    } catch (const std::invalid_argument& ia) {
                        cout << "Warning: Could not parse attribute '" << attributeName << "' as an integer." << endl;
                        parsingFailure = true;
                    }
                }
                
                if (parsingFailure) {
                    // FALL BACK TO STRING
                    attribute = outputGroup.createAttribute(attributeName, types.strType, attributeDataSpace);
                    attribute.write(types.strType, attributeValue);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    string inputFileName;
    string outputFileName;
    bool slow;
    
    if (!getOptions(argc, argv, inputFileName, outputFileName, slow)) {
        return 1;
    }
    
    cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << endl;
    
    auto tStart = chrono::high_resolution_clock::now();

    fitsfile* inputFilePtr;
        
    if (!openFitsFile(inputFileName, inputFilePtr)) {
        return 1;
    }
        
    int N;
    long stokes, depth, height, width;
    
    if (!getDims(inputFilePtr, N, stokes, depth, height, width)) {
        return 1;
    }
            
    map<string, vector<hsize_t>> dataDims;
    map<string, int> numBins;
    getDataDims(N, stokes, depth, height, width, dataDims, numBins);

    int status = 0;
    DataSpace swizzledDataSpace(N, dataDims["swizzled"].data());
    DataSpace standardDataSpace(N, dataDims["standard"].data());
    
    DSetCreatPropList standardCreatePlist;
    standardCreatePlist.setChunk(N, dataDims["tile"].data());
    
    string tempOutputFileName = outputFileName + ".tmp";
    H5File outputFile(tempOutputFileName, H5F_ACC_TRUNC);
    Group outputGroup = outputFile.createGroup("0");

    Types types;

    writeHeaders(inputFilePtr, types, outputGroup);

    if (depth > 1) {
        auto swizzledGroup = outputGroup.createGroup("SwizzledData");
        string swizzledName = N == 3 ? "ZYX" : "ZYXW";
        auto swizzledDataSet = swizzledGroup.createDataSet(swizzledName, types.floatType, swizzledDataSpace);
    }

    auto standardDataSet = outputGroup.createDataSet("DATA", types.floatType, standardDataSpace, standardCreatePlist);

    auto cubeSize = depth * height * width;
    cout << "Allocating " << cubeSize * 4 * 2 * 1e-9 << " GB of memory..." << flush;
    auto tStartAlloc = chrono::high_resolution_clock::now();

    float* standardCube = new float[cubeSize];
    
    vector<float> minValsXY(depth * stokes);
    vector<float> maxValsXY(depth * stokes);
    vector<float> sumsXY(depth * stokes);
    vector<float> sumsSqXY(depth * stokes);
    vector<int64_t> nanCountsXY(depth * stokes);
    
    vector<int64_t> histogramsXY(depth * stokes * numBins["XY"]);
    
    float* rotatedCube;
    
    vector<float> minValsZ;
    vector<float> maxValsZ;
    vector<float> sumsZ;
    vector<float> sumsSqZ;
    vector<int64_t> nanCountsZ;

    vector<float> minValsXYZ;
    vector<float> maxValsXYZ;
    vector<float> sumsXYZ;
    vector<float> sumsSqXYZ;
    vector<int64_t> nanCountsXYZ;
    
    // XYZ histograms calculated using OpenMP and summed later
    vector<int64_t> partialHistogramsXYZ;
    vector<int64_t> histogramsXYZ;
    
    if (depth > 1) {
        rotatedCube = new float[cubeSize];
        
        minValsZ.resize(width * height * stokes, numeric_limits<float>::max());
        maxValsZ.resize(width * height * stokes, -numeric_limits<float>::max());
        sumsZ.resize(width * height * stokes);
        sumsSqZ.resize(width * height * stokes);
        nanCountsZ.resize(width * height * stokes);

        minValsXYZ.resize(stokes);
        maxValsXYZ.resize(stokes);
        sumsXYZ.resize(stokes);
        sumsSqXYZ.resize(stokes);
        nanCountsXYZ.resize(stokes);
        
        // XYZ histograms calculated using OpenMP and summed later
        partialHistogramsXYZ.resize(depth * stokes * numBins["XYZ"]);
        histogramsXYZ.resize(stokes * numBins["XYZ"]);
    }

    auto tEndAlloc = chrono::high_resolution_clock::now();
    auto dtAlloc = chrono::duration_cast<chrono::milliseconds>(tEndAlloc - tStartAlloc).count();
    cout << "Done in " << dtAlloc * 1e-3 << " seconds" << endl;

    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        // Read data into memory space
        hsize_t memDims[] = {depth, height, width};
        DataSpace memspace(3, memDims);

        cout << "Reading Stokes " << currentStokes << " dataset..." << flush;
        auto tStartRead = chrono::high_resolution_clock::now();
        long fpixel[] = {1, 1, 1, currentStokes + 1};
        fits_read_pix(inputFilePtr, TFLOAT, fpixel, cubeSize, NULL, standardCube, NULL, &status);
        auto tEndRead = chrono::high_resolution_clock::now();
        auto dtRead = chrono::duration_cast<chrono::milliseconds>(tEndRead - tStartRead).count();
        float readSpeed = (cubeSize * 4) * 1.0e-6 / (dtRead * 1.0e-3);
        cout << "Done in " << dtRead * 1e-3 << " seconds (" << readSpeed << " MB/s)" << endl;

        cout << "Processing Stokes " << currentStokes << " dataset..." << flush;
        auto tStartProcess = chrono::high_resolution_clock::now();

        // First loop calculates stats for each XY slice and rotates the dataset
#pragma omp parallel for
        for (auto i = 0; i < depth; i++) {
            float minVal = numeric_limits<float>::max();
            float maxVal = -numeric_limits<float>::max();
            float sum = 0;
            float sumSq = 0;
            int64_t nanCount = 0;

            for (auto j = 0; j < height; j++) {
                for (auto k = 0; k < width; k++) {
                    auto sourceIndex = k + width * j + (height * width) * i;
                    auto destIndex = i + depth * j + (height * depth) * k;
                    auto val = standardCube[sourceIndex];
                    
                    if (depth > 1) {
                        rotatedCube[destIndex] = val;
                    }
                    
                    // Stats
                    if (!isnan(val)) {
                        // This should be safe. It would only fail if we had a strictly descending or ascending sequence;
                        // very unlikely when we're iterating over all values in a channel.
                        if (val < minVal) {
                            minVal = val;
                        } else if (val > maxVal) {
                            maxVal = val;
                        }
                        sum += val;
                        sumSq += val * val;
                    } else {
                        nanCount += 1;
                    }
                }
            }

            auto indexXY = currentStokes * depth + i;
            
            nanCountsXY[indexXY] = nanCount;
            
            if (nanCount != (height * width)) {
                minValsXY[indexXY] = minVal;
                maxValsXY[indexXY] = maxVal;
                sumsXY[indexXY] = sum;
                sumsSqXY[indexXY] = sumSq;
            } else {
                minValsXY[indexXY] = NAN;
                maxValsXY[indexXY] = NAN;
                sumsXY[indexXY] = NAN;
                sumsSqXY[indexXY] = NAN;
            }
        }
        
        double xyzMin;
        double xyzMax;

        if (depth > 1) {
            // Consolidate XY stats into XYZ stats
            double xyzSum = 0;
            double xyzSumSq = 0;
            int64_t xyzNanCount = 0;
            xyzMin = minValsXY[currentStokes * depth];
            xyzMax = maxValsXY[currentStokes * depth];

            for (auto i = 0; i < depth; i++) {
                auto indexXY = currentStokes * depth + i;
                auto sum = sumsXY[indexXY];
                if (!isnan(sum)) {
                    xyzSum += sum;
                    xyzSumSq += sumsSqXY[indexXY];
                    xyzMin = fmin(xyzMin, minValsXY[indexXY]);
                    xyzMax = fmax(xyzMax, maxValsXY[indexXY]);
                }
                xyzNanCount += nanCountsXY[indexXY];
            }

            sumsXYZ[currentStokes] = xyzSum;
            sumsSqXYZ[currentStokes] = xyzSumSq;
            minValsXYZ[currentStokes] = xyzMin;
            maxValsXYZ[currentStokes] = xyzMax;
            nanCountsXYZ[currentStokes] = xyzNanCount;
        }

        cout << "1..." << flush;

        if (depth > 1) {
            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
#pragma omp parallel for
            for (auto j = 0; j < height; j++) {
                for (auto k = 0; k < width; k++) {
                    float minVal = numeric_limits<float>::max();
                    float maxVal = -numeric_limits<float>::max();
                    float sum = 0;
                    float sumSq = 0;
                    int64_t nanCount = 0;
                    
                    for (auto i = 0; i < depth; i++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto val = standardCube[sourceIndex];

                        if (!isnan(val)) {
                            // Not replacing this with if/else; too much risk of encountering an ascending / descending sequence.
                            minVal = min(minVal, val);
                            maxVal = max(maxVal, val);
                            sum += val;
                            sumSq += val * val;
                        } else {
                            nanCount += 1;
                        }
                    }
                    
                    auto indexZ = currentStokes * width * height + k + j * width;
                    
                    nanCountsZ[indexZ] = nanCount;
                    
                    if (nanCount != (height * width)) {
                        minValsZ[indexZ] = minVal;
                        maxValsZ[indexZ] = maxVal;
                        sumsZ[indexZ] = sum;
                        sumsSqZ[indexZ] = sumSq;
                    } else {
                        minValsZ[indexZ] = NAN;
                        maxValsZ[indexZ] = NAN;
                        sumsZ[indexZ] = NAN;
                        sumsSqZ[indexZ] = NAN;
                    }
                }
            }
        }
        
        cout << "2..." << flush;

        // Third loop handles histograms
        
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
            double sliceMin = minValsXY[indexXY];
            double sliceMax = maxValsXY[indexXY];
            double range = sliceMax - sliceMin;

            if (isnan(sliceMin) || isnan(sliceMax) || range == 0) {
                continue;
            }

            for (auto j = 0; j < width * height; j++) {
                auto val = standardCube[i * width * height + j];

                if (!isnan(val)) {
                    // XY Histogram
                    int binIndex = min(numBins["XY"] - 1, (int)(numBins["XY"] * (val - sliceMin) / range));
                    histogramsXY[currentStokes * depth * numBins["XY"] + i * numBins["XY"] + binIndex]++;
                    
                    if (depth > 1) {
                        // XYZ Partial histogram
                        int binIndexXYZ = min(numBins["XYZ"] - 1, (int)(numBins["XYZ"] * (val - cubeMin) / cubeRange));
                        partialHistogramsXYZ[currentStokes * depth * numBins["XYZ"] + i * numBins["XYZ"] + binIndexXYZ]++;
                    }
                }
            }
        }
        
        if (depth > 1) {
            // Consolidate partial XYZ histograms into final histogram
            for (auto i = 0; i < depth; i++) {
                for (auto j = 0; j < numBins["XYZ"]; j++) {
                    histogramsXYZ[currentStokes * numBins["XYZ"] + j] +=
                        partialHistogramsXYZ[currentStokes * depth * numBins["XYZ"] + i * numBins["XYZ"] + j];
                }
            }
        }

        auto tEndRotate = chrono::high_resolution_clock::now();
        auto dtRotate = chrono::duration_cast<chrono::milliseconds>(tEndRotate - tStartProcess).count();
        float rotateSpeed = (cubeSize * 4) * 1.0e-6 / (dtRotate * 1.0e-3);
        cout << "Done in " << dtRotate * 1e-3 << " seconds" << endl;

        cout << "Writing Stokes " << currentStokes << " dataset..." << flush;
        auto tStartWrite = chrono::high_resolution_clock::now();

        auto sliceDataSpace = standardDataSet.getSpace();
        vector<hsize_t> count = {height, width};
        vector<hsize_t> start = {0, 0};

        if (N >= 3) {
            count.insert(count.begin(), depth);
            start.insert(start.begin(), 0);
        }
        if (N == 4) {
            count.insert(count.begin(), 1);
            start.insert(start.begin(), currentStokes);
        }

        sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

        standardDataSet.write(standardCube, PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
        if (depth > 1) {
            auto swizzledGroup = outputGroup.openGroup("SwizzledData");
            string swizzledName = N == 3 ? "ZYX" : "ZYXW";
            auto swizzledDataSet = swizzledGroup.openDataSet(swizzledName);
            swizzledDataSet.write(rotatedCube, PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
        }

        auto tEndWrite = chrono::high_resolution_clock::now();
        auto dtWrite = chrono::duration_cast<chrono::milliseconds>(tEndWrite - tStartWrite).count();
        float writeSpeed = (2 * cubeSize * 4) * 1.0e-6 / (dtWrite * 1.0e-3);
        cout << "Done in " << dtWrite * 1e-3 << " seconds (" << writeSpeed << " MB/s)" << endl;
    }

    auto statsGroup = outputGroup.createGroup("Statistics");
    auto statsXYGroup = statsGroup.createGroup("XY");
    
    DataSpace xyStatsDataSpace(N - 2, dataDims["xyStats"].data());
    DataSpace xyHistogramDataSpace(N - 1, dataDims["xyHistogram"].data());
    
    auto xyMinDataSet = statsXYGroup.createDataSet("MIN", types.floatType, xyStatsDataSpace);
    auto xyMaxDataSet = statsXYGroup.createDataSet("MAX", types.floatType, xyStatsDataSpace);
    auto xySumDataSet = statsXYGroup.createDataSet("SUM", types.floatType, xyStatsDataSpace);
    auto xySumSqDataSet = statsXYGroup.createDataSet("SUM_SQ", types.floatType, xyStatsDataSpace);
    auto xyNanCountDataSet = statsXYGroup.createDataSet("NAN_COUNT", types.intType, xyStatsDataSpace);
    auto xyHistogramDataSet = statsXYGroup.createDataSet("HISTOGRAM", types.intType, xyHistogramDataSpace);

    xyMinDataSet.write(minValsXY.data(), PredType::NATIVE_FLOAT);
    xyMaxDataSet.write(maxValsXY.data(), PredType::NATIVE_FLOAT);
    xySumDataSet.write(sumsXY.data(), PredType::NATIVE_FLOAT);
    xySumSqDataSet.write(sumsSqXY.data(), PredType::NATIVE_FLOAT);
    xyNanCountDataSet.write(nanCountsXY.data(), PredType::NATIVE_INT64);
    xyHistogramDataSet.write(histogramsXY.data(), PredType::NATIVE_INT64);

    if (depth > 1) {
        auto statsXYZGroup = statsGroup.createGroup("XYZ");
        
        DataSpace xyzStatsDataSpace(N - 3, dataDims["xyStats"].data());
        DataSpace xyzHistogramDataSpace(N - 2, dataDims["xyzHistogram"].data());
        
        auto xyzMinDataSet = statsXYZGroup.createDataSet("MIN", types.floatType, xyzStatsDataSpace);
        auto xyzMaxDataSet = statsXYZGroup.createDataSet("MAX", types.floatType, xyzStatsDataSpace);
        auto xyzSumDataSet = statsXYZGroup.createDataSet("SUM", types.floatType, xyzStatsDataSpace);
        auto xyzSumSqDataSet = statsXYZGroup.createDataSet("SUM_SQ", types.floatType, xyzStatsDataSpace);
        auto xyzNanCountDataSet = statsXYZGroup.createDataSet("NAN_COUNT", types.intType, xyzStatsDataSpace);
        auto xyzHistogramDataSet = statsXYZGroup.createDataSet("HISTOGRAM", types.intType, xyzHistogramDataSpace);
        
        xyzMinDataSet.write(minValsXYZ.data(), PredType::NATIVE_FLOAT);
        xyzMaxDataSet.write(maxValsXYZ.data(), PredType::NATIVE_FLOAT);
        xyzSumDataSet.write(sumsXYZ.data(), PredType::NATIVE_FLOAT);
        xyzSumSqDataSet.write(sumsSqXYZ.data(), PredType::NATIVE_FLOAT);
        xyzNanCountDataSet.write(nanCountsXYZ.data(), PredType::NATIVE_INT64);
        xyzHistogramDataSet.write(histogramsXYZ.data(), PredType::NATIVE_INT64);

        auto statsZGroup = statsGroup.createGroup("Z");
        
        DataSpace zStatsDataSpace(N - 1, dataDims["zStats"].data());
        
        auto zMinDataSet = statsZGroup.createDataSet("MIN", types.floatType, zStatsDataSpace);
        auto zMaxDataSet = statsZGroup.createDataSet("MAX", types.floatType, zStatsDataSpace);
        auto zSumDataSet = statsZGroup.createDataSet("SUM", types.floatType, zStatsDataSpace);
        auto zSumSqDataSet = statsZGroup.createDataSet("SUM_SQ", types.floatType, zStatsDataSpace);
        auto zNanCountDataSet = statsZGroup.createDataSet("NAN_COUNT", types.intType, zStatsDataSpace);

        zMinDataSet.write(minValsZ.data(), PredType::NATIVE_FLOAT);
        zMaxDataSet.write(maxValsZ.data(), PredType::NATIVE_FLOAT);
        zSumDataSet.write(sumsZ.data(), PredType::NATIVE_FLOAT);
        zSumSqDataSet.write(sumsSqZ.data(), PredType::NATIVE_FLOAT);
        zNanCountDataSet.write(nanCountsZ.data(), PredType::NATIVE_INT64);
    }

    outputFile.close();
    rename(tempOutputFileName.c_str(), outputFileName.c_str());

    auto tEnd = chrono::high_resolution_clock::now();
    auto dtTotal = chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count();
    cout << "FITS file converted in " << dtTotal * 1e-3 << " seconds" << endl;
    delete[] standardCube;
    
    if (depth > 1) {
        delete[] rotatedCube;
    }
    return 0;
}
