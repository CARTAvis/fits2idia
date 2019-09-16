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

#define TILE_SIZE 512

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
    
    void writeDset(Group& group, string name, vector<float>& vals, FloatType type, DataSpace dspace) {
        auto dset = group.createDataSet(name, type, dspace);
        dset.write(vals.data(), PredType::NATIVE_FLOAT);
    }
    
    void writeDset(Group& group, string name, vector<int64_t>& vals, IntType type, DataSpace dspace) {
        auto dset = group.createDataSet(name, type, dspace);
        dset.write(vals.data(), PredType::NATIVE_INT64);
    }
    
    void write(Group& group, DataSpace statsSpace, FloatType floatType, IntType intType) {
        writeDset(group, "MIN", minVals, floatType, statsSpace);
        writeDset(group, "MAX", maxVals, floatType, statsSpace);
        writeDset(group, "SUM", sums, floatType, statsSpace);
        writeDset(group, "SUM_SQ", sumsSq, floatType, statsSpace);
        
        writeDset(group, "NAN_COUNT", nanCounts, intType, statsSpace);
    }
    
    void write(Group& group, DataSpace statsSpace, DataSpace histSpace, FloatType floatType, IntType intType) {
        write(group, statsSpace, floatType, intType);
        
        writeDset(group, "HISTOGRAM", histograms, intType, histSpace);
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

class Image {
public:
    Image() {}
    
    Image(string inputFileName, string outputFileName, bool slow) :
        tileDims({TILE_SIZE, TILE_SIZE}),
        status(0),
        strType(PredType::C_S1, 256),
        boolType(PredType::NATIVE_HBOOL), 
        doubleType(PredType::NATIVE_DOUBLE),
        floatType(PredType::NATIVE_FLOAT),
        intType(PredType::NATIVE_INT64)
    {        
        fits_open_file(&inputFilePtr, inputFileName.c_str(), READONLY, &status);
        
        if (status != 0) {
            throw "error opening FITS file";
        }
        
        int bitpix;
        fits_get_img_type(inputFilePtr, &bitpix, &status);

        if (bitpix != -32) {
            throw "Currently only supports FP32 files";
        }
        
        fits_get_img_dim(inputFilePtr, &N, &status);
    
        if (N < 2 || N > 4) {
            throw "Currently only supports 2D, 3D and 4D cubes";
        }
        
        long dims[4];
        fits_get_img_size(inputFilePtr, 4, dims, &status);
            
        stokes = N == 4 ? dims[3] : 1;
        depth = N >= 3 ? dims[2] : 1;
        height = dims[1];
        width = dims[0];
        
        numBinsXY = int(std::max(sqrt(width * height), 2.0));
        numBinsXYZ = numBinsXY;
        
        this->slow = slow;
        
        // Customise types
        doubleType.setOrder(H5T_ORDER_LE);
        floatType.setOrder(H5T_ORDER_LE);
        intType.setOrder(H5T_ORDER_LE);
        
        this->outputFileName = outputFileName;
        tempOutputFileName = outputFileName + ".tmp";
        
        outputFile = H5File(tempOutputFileName, H5F_ACC_TRUNC);
        outputGroup = outputFile.createGroup("0");
        
        // TODO: I'm sure there's a way to avoid doing some of this and to remove some of the duplication with the stats size

            
            if (N == 2) {
                standardDims = {height, width};
                swizzledDims = {width, height};
                xyStatsDims = {};
                xyHistogramDims = {numBinsXY};
            } else if (N == 3) {
                standardDims = {depth, height, width};
                swizzledDims = {width, height, depth};
                xyStatsDims = {depth};
                xyHistogramDims = {depth, numBinsXY};
                xyzStatsDims = {};
                xyzHistogramDims = {numBinsXYZ};
                zStatsDims = {height, width};
                
            } else if (N == 4) {
                standardDims = {stokes, depth, height, width};
                swizzledDims = {stokes, width, height, depth};
                xyStatsDims = {stokes, depth};
                xyHistogramDims = {stokes, depth, numBinsXY};
                xyzStatsDims = {stokes};
                xyzHistogramDims = {stokes, numBinsXYZ};
                zStatsDims = {stokes, height, width};
            }
    }
    
    ~Image() {
        outputFile.close();
    }
    
    void copyHeaders() {
        DataSpace attributeDataSpace(H5S_SCALAR);
        
        Attribute attribute = outputGroup.createAttribute("SCHEMA_VERSION", strType, attributeDataSpace);
        attribute.write(strType, SCHEMA_VERSION);
        attribute = outputGroup.createAttribute("HDF5_CONVERTER", strType, attributeDataSpace);
        attribute.write(strType, HDF5_CONVERTER);
        attribute = outputGroup.createAttribute("HDF5_CONVERTER_VERSION", strType, attributeDataSpace);
        attribute.write(strType, HDF5_CONVERTER_VERSION);

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
                            cout << "Warning: Could not parse attribute '" << attributeName << "' as a float." << endl;
                            parsingFailure = true;
                        }
                    } else {
                        // TRY TO PARSE AS INTEGER
                        try {
                            int64_t attributeValueInt = std::stoi(attributeValue);
                            attribute = outputGroup.createAttribute(attributeName, intType, attributeDataSpace);
                            attribute.write(intType, &attributeValueInt);
                        } catch (const std::invalid_argument& ia) {
                            cout << "Warning: Could not parse attribute '" << attributeName << "' as an integer." << endl;
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
    
    void copyData() {
        auto cubeSize = depth * height * width;
        cout << "Allocating " << cubeSize * 4 * 2 * 1e-9 << " GB of memory..." << flush;
        auto tStartAlloc = chrono::high_resolution_clock::now();

        standardCube = new float[cubeSize];
        
        statsXY = Stats(depth * stokes, numBinsXY);
        
        if (depth > 1) {
            if (!slow) {
                rotatedCube = new float[cubeSize];
            }
            statsZ = Stats(width * height * stokes);
            statsXYZ = Stats(stokes, numBinsXYZ, depth);
        }

        auto tEndAlloc = chrono::high_resolution_clock::now();
        auto dtAlloc = chrono::duration_cast<chrono::milliseconds>(tEndAlloc - tStartAlloc).count();
        cout << "Done in " << dtAlloc * 1e-3 << " seconds" << endl;
            
        DSetCreatPropList standardCreatePlist;
        standardCreatePlist.setChunk(N, tileDims.data());
        auto standardDataSpace = DataSpace(N, standardDims.data());
        
        if (depth > 1) {
            auto swizzledGroup = outputGroup.createGroup("SwizzledData");
            string swizzledName = N == 3 ? "ZYX" : "ZYXW";
            auto swizzledDataSpace = DataSpace(N, swizzledDims.data());
            auto swizzledDataSet = swizzledGroup.createDataSet(swizzledName, floatType, swizzledDataSpace);
        }

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
                        
                        if (depth > 1 && !slow) {
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
            
            auto standardDataSet = outputGroup.createDataSet("DATA", floatType, standardDataSpace, standardCreatePlist);
            auto sliceDataSpace = standardDataSet.getSpace();
            
            vector<hsize_t> count;
            vector<hsize_t> start;
            
            if (N == 2) {
                count = {height, width};
                start = {0, 0};
            } else if (N >= 3) {
                count = {depth, height, width};
                start = {0, 0, 0};
            } else if (N == 4) {
                count = {1, depth, height, width};
                start = {currentStokes, 0, 0, 0};
            }

            sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

            standardDataSet.write(standardCube, PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
            
            if (depth > 1 && !slow) {
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
        
        auto xyStatsDataSpace = DataSpace(N - 2, xyStatsDims.data());
        auto xyHistogramDataSpace = DataSpace(N - 1, xyHistogramDims.data());
        auto statsXYGroup = statsGroup.createGroup("XY"); 
        statsXY.write(statsXYGroup, xyStatsDataSpace, xyHistogramDataSpace, floatType, intType);

        if (depth > 1) {
            auto xyzStatsDataSpace = DataSpace(N - 3, xyzStatsDims.data());
            auto xyzHistogramDataSpace = DataSpace(N - 2, xyzHistogramDims.data());
            auto zStatsDataSpace = DataSpace(N - 1, zStatsDims.data());
            auto statsXYZGroup = statsGroup.createGroup("XYZ"); 
            auto statsZGroup = statsGroup.createGroup("Z"); 
            statsXYZ.write(statsXYZGroup, xyzStatsDataSpace, xyzHistogramDataSpace, floatType, intType);
            statsZ.write(statsZGroup, zStatsDataSpace, floatType, intType);
        }
        
        delete[] standardCube;
        
        if (depth > 1 && !slow) {
            delete[] rotatedCube;
        }
        
        // call slow swizzle function if necessary
        if (depth > 1 && slow) {
            slowSwizzle();
        }
        
        // right at the end
        rename(tempOutputFileName.c_str(), outputFileName.c_str());
    }
    
    void slowSwizzle() {
        // alternative function for swizzling
    }
    
    
private:
    string tempOutputFileName;
    string outputFileName;
    fitsfile* inputFilePtr;
    H5File outputFile;
    Group outputGroup;
    
    float* standardCube;
    float* rotatedCube;
    
    // Stats
    Stats statsXY;
    Stats statsZ;
    Stats statsXYZ;
    
    int status;
    bool slow;
    
    int N;
    long stokes, depth, height, width;
    int numBinsXY, numBinsXYZ;

    // Dimensions
    vector<hsize_t> standardDims;
    vector<hsize_t> swizzledDims;
    vector<hsize_t> xyStatsDims;
    vector<hsize_t> xyHistogramDims;
    vector<hsize_t> xyzStatsDims;
    vector<hsize_t> xyzHistogramDims;
    vector<hsize_t> zStatsDims;
    vector<hsize_t> tileDims;
    
    // Types
    StrType strType;
    IntType boolType;
    FloatType doubleType;
    FloatType floatType;
    IntType intType;
};

int main(int argc, char** argv) {
    string inputFileName;
    string outputFileName;
    bool slow;
    
    if (!getOptions(argc, argv, inputFileName, outputFileName, slow)) {
        return 1;
    }
    
    Image image;
        
    try {
        image = Image(inputFileName, outputFileName, slow);
    } catch (const char* msg) {
        cerr << msg << endl;
        return 1;
    }
    
    cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << endl;
    
    auto tStart = chrono::high_resolution_clock::now();
    
    image.copyHeaders();
    image.copyData();
    
    auto tEnd = chrono::high_resolution_clock::now();
    auto dtTotal = chrono::duration_cast<chrono::milliseconds>(tEnd - tStart).count();
    cout << "FITS file converted in " << dtTotal * 1e-3 << " seconds" << endl;

    return 0;
}
