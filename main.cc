#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <getopt.h>

#include <H5Cpp.h>
#include <fitsio.h>

using namespace H5;
using namespace std;

#define SCHEMA_VERSION "0.2"
#define HDF5_CONVERTER "hdf_convert"
#define HDF5_CONVERTER_VERSION "0.1.8"

// TODO: this is messy and should be eliminated
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

// TODO: this should also incorporate stats vector sizes and dataspace dimensions
struct DataDims {    
    DataDims(long height, long width, int numBinsXY, int numBinsXYZ, int tileSize) : 
        standard({height, width}), 
        swizzled({width, height}), 
        xyHistogram({numBinsXY}), 
        zStats({height, width}), 
        xyzHistogram({numBinsXYZ}), 
        tile({tileSize, tileSize}) 
    {}
    
    DataDims(long depth, long height, long width, int numBinsXY, int numBinsXYZ, int tileSize) : 
        standard({depth, height, width}), 
        swizzled({width, height, depth}), 
        xyHistogram({depth, numBinsXY}), 
        xyStats({depth}), 
        zStats({height, width}), 
        xyzHistogram({numBinsXYZ}), 
        tile({1, tileSize, tileSize}) 
    {}
    
    DataDims(long stokes, long depth, long height, long width, int numBinsXY, int numBinsXYZ, int tileSize) : 
        standard({stokes, depth, height, width}), 
        swizzled({stokes, width, height, depth}), 
        xyHistogram({stokes, depth, numBinsXY}), 
        xyStats({stokes, depth}), 
        zStats({stokes, height, width}), 
        xyzHistogram({stokes, numBinsXYZ}), 
        xyzStats({stokes}), 
        tile({1, 1, tileSize, tileSize}) 
    {}
    
    static DataDims getDataDims(int N, long stokes, long depth, long height, long width, int numBinsXY, int numBinsXYZ, int tileSize)
    {
        if (N == 2) {
            return DataDims(height, width, numBinsXY, numBinsXYZ, tileSize);
        } else if (N == 3) {
            return DataDims(depth, height, width, numBinsXY, numBinsXYZ, tileSize);
        } else if (N == 4) {
            return DataDims(stokes, depth, height, width, numBinsXY, numBinsXYZ, tileSize);
        }
    }

    vector<hsize_t> standard;
    vector<hsize_t> swizzled;
    vector<hsize_t> xyHistogram;
    vector<hsize_t> xyStats;
    vector<hsize_t> zStats;
    vector<hsize_t> xyzHistogram;
    vector<hsize_t> xyzStats;
    vector<hsize_t> tile;
};

struct Stats {
    Stats() {}
    
    Stats(long size, int numBins=0, long partialHistMult=0) : 
        minVals(size, numeric_limits<float>::max()), 
        maxVals(size, -numeric_limits<float>::max()),
        sums(size),
        sumsSq(size),
        nanCounts(size),
        partialHistograms(size * numBins * partialHistMult),
        histograms(size * numBins)
    {}
    
    void writeDset(Group group, string name, vector<float> vals, FloatType type, DataSpace dspace) {
        auto dset = group.createDataSet(name, type, dspace);
        dset.write(vals.data(), PredType::NATIVE_FLOAT);
    }
    
    void writeDset(Group group, string name, vector<int64_t> vals, IntType type, DataSpace dspace) {
        auto dset = group.createDataSet(name, type, dspace);
        dset.write(vals.data(), PredType::NATIVE_INT64);
    }
    
    void write(Group group, Types types, DataSpace statsDspace) {
        writeDset(group, "MIN", minVals, types.floatType, statsDspace);
        writeDset(group, "MAX", maxVals, types.floatType, statsDspace);
        writeDset(group, "SUM", sums, types.floatType, statsDspace);
        writeDset(group, "SUM_SQ", sumsSq, types.floatType, statsDspace);
        
        writeDset(group, "NAN_COUNT", nanCounts, types.intType, statsDspace);
    }
    
    void write(Group group, Types types, DataSpace statsDspace, DataSpace histDspace) {
        write(group, types, statsDspace);
        
        writeDset(group, "HISTOGRAM", histograms, types.intType, histDspace);
    }

    vector<float> minVals;
    vector<float> maxVals;
    vector<float> sums;
    vector<float> sumsSq;
    vector<int64_t> nanCounts;
    vector<int64_t> partialHistograms;
    vector<int64_t> histograms;
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
    
    int numBinsXY = int(std::max(sqrt(width * height), 2.0));
    int numBinsXYZ = numBinsXY;
    int tileSize = 512;
            
    DataDims dataDims = DataDims::getDataDims(N, stokes, depth, height, width, numBinsXY, numBinsXYZ, tileSize);

    int status = 0;
    DataSpace swizzledDataSpace(N, dataDims.swizzled.data());
    DataSpace standardDataSpace(N, dataDims.standard.data());
    
    DSetCreatPropList standardCreatePlist;
    standardCreatePlist.setChunk(N, dataDims.tile.data());
    
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
    float* rotatedCube;
    
    Stats statsXY = Stats(depth * stokes, numBinsXY);
    Stats statsZ;
    Stats statsXYZ;
    
    if (depth > 1) {
        rotatedCube = new float[cubeSize];
        statsZ = Stats(width * height * stokes);
        statsXYZ = Stats(stokes, numBinsXYZ, depth);
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
            
            statsXY.nanCounts[indexXY] = nanCount;
            
            if (nanCount != (height * width)) {
                statsXY.minVals[indexXY] = minVal;
                statsXY.maxVals[indexXY] = maxVal;
                statsXY.sums[indexXY] = sum;
                statsXY.sumsSq[indexXY] = sumSq;
            } else {
                statsXY.minVals[indexXY] = NAN;
                statsXY.maxVals[indexXY] = NAN;
                statsXY.sums[indexXY] = NAN;
                statsXY.sumsSq[indexXY] = NAN;
            }
        }
        
        double xyzMin;
        double xyzMax;

        if (depth > 1) {
            // Consolidate XY stats into XYZ stats
            double xyzSum = 0;
            double xyzSumSq = 0;
            int64_t xyzNanCount = 0;
            xyzMin = statsXY.minVals[currentStokes * depth];
            xyzMax = statsXY.maxVals[currentStokes * depth];

            for (auto i = 0; i < depth; i++) {
                auto indexXY = currentStokes * depth + i;
                auto sum = statsXY.sums[indexXY];
                if (!isnan(sum)) {
                    xyzSum += sum;
                    xyzSumSq += statsXY.sumsSq[indexXY];
                    xyzMin = fmin(xyzMin, statsXY.minVals[indexXY]);
                    xyzMax = fmax(xyzMax, statsXY.maxVals[indexXY]);
                }
                xyzNanCount += statsXY.nanCounts[indexXY];
            }

            statsXYZ.sums[currentStokes] = xyzSum;
            statsXYZ.sumsSq[currentStokes] = xyzSumSq;
            statsXYZ.minVals[currentStokes] = xyzMin;
            statsXYZ.maxVals[currentStokes] = xyzMax;
            statsXYZ.nanCounts[currentStokes] = xyzNanCount;
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
                    
                    statsZ.nanCounts[indexZ] = nanCount;
                    
                    if (nanCount != (height * width)) {
                        statsZ.minVals[indexZ] = minVal;
                        statsZ.maxVals[indexZ] = maxVal;
                        statsZ.sums[indexZ] = sum;
                        statsZ.sumsSq[indexZ] = sumSq;
                    } else {
                        statsZ.minVals[indexZ] = NAN;
                        statsZ.maxVals[indexZ] = NAN;
                        statsZ.sums[indexZ] = NAN;
                        statsZ.sumsSq[indexZ] = NAN;
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
            double sliceMin = statsXY.minVals[indexXY];
            double sliceMax = statsXY.maxVals[indexXY];
            double range = sliceMax - sliceMin;

            if (isnan(sliceMin) || isnan(sliceMax) || range == 0) {
                continue;
            }

            for (auto j = 0; j < width * height; j++) {
                auto val = standardCube[i * width * height + j];

                if (!isnan(val)) {
                    // XY Histogram
                    int binIndex = min(numBinsXY - 1, (int)(numBinsXY * (val - sliceMin) / range));
                    statsXY.histograms[currentStokes * depth * numBinsXY + i * numBinsXY + binIndex]++;
                    
                    if (depth > 1) {
                        // XYZ Partial histogram
                        int binIndexXYZ = min(numBinsXYZ - 1, (int)(numBinsXYZ * (val - cubeMin) / cubeRange));
                        statsXYZ.partialHistograms[currentStokes * depth * numBinsXYZ + i * numBinsXYZ + binIndexXYZ]++;
                    }
                }
            }
        }
        
        if (depth > 1) {
            // Consolidate partial XYZ histograms into final histogram
            for (auto i = 0; i < depth; i++) {
                for (auto j = 0; j < numBinsXYZ; j++) {
                    statsXYZ.histograms[currentStokes * numBinsXYZ + j] +=
                        statsXYZ.partialHistograms[currentStokes * depth * numBinsXYZ + i * numBinsXYZ + j];
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
    DataSpace xyStatsDataSpace(N - 2, dataDims.xyStats.data());
    DataSpace xyHistogramDataSpace(N - 1, dataDims.xyHistogram.data());
    statsXY.write(statsXYGroup, types, xyStatsDataSpace, xyHistogramDataSpace);

    if (depth > 1) {
        auto statsXYZGroup = statsGroup.createGroup("XYZ");
        DataSpace xyzStatsDataSpace(N - 3, dataDims.xyzStats.data());
        DataSpace xyzHistogramDataSpace(N - 2, dataDims.xyzHistogram.data());
        statsXYZ.write(statsXYZGroup, types, xyzStatsDataSpace, xyzHistogramDataSpace);

        auto statsZGroup = statsGroup.createGroup("Z");
        DataSpace zStatsDataSpace(N - 1, dataDims.zStats.data());
        statsZ.write(statsZGroup, types, zStatsDataSpace);
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
