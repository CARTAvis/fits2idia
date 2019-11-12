#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <getopt.h>

#include <H5Cpp.h>
#include <fitsio.h>

using namespace H5;
using namespace std;

#define SCHEMA_VERSION "0.2"
#define HDF5_CONVERTER "hdf_convert"
#define HDF5_CONVERTER_VERSION "0.1.9beta2"

hsize_t TILE_SIZE = 512;
hsize_t MIN_MIPMAP_SIZE = 128;

struct StatsDims {
    StatsDims() {}
    
    StatsDims(vector<hsize_t> statsDims, hsize_t statsSize, int numBins=0, hsize_t partialHistMult=0) :
        statsDims(statsDims),
        statsSize(statsSize),
        histSize(statsSize * numBins),
        partialHistSize(statsSize * numBins * partialHistMult),
        numBins(numBins)
    {
        if (numBins) {
            for (auto d : statsDims) {
                histDims.push_back(d);
            }
            histDims.push_back(numBins);
        }
    }
    
    vector<hsize_t> statsDims;
    vector<hsize_t> histDims;
    hsize_t statsSize;
    hsize_t histSize;
    hsize_t partialHistSize;
    int numBins;
};

// This is a bit repetitive but I think it's best to define each case explicitly
struct Dims {
    Dims() {}
    
    Dims(hsize_t width, hsize_t height) :
        N(2),
        width(width), height(height), depth(1), stokes(1),
        standard({height, width}),
        mipMapExtra({}),
        tileDims({TILE_SIZE, TILE_SIZE}),
        statsXY({}, depth * stokes, defaultBins()) // dims, size, bins
    {}
    
    Dims(hsize_t width, hsize_t height, hsize_t depth) :
        N(3),
        width(width), height(height), depth(depth), stokes(1),
        standard({depth, height, width}),
        swizzled({width, height, depth}),
        mipMapExtra({depth}),
        tileDims({1, TILE_SIZE, TILE_SIZE}),
        statsXY({depth}, depth * stokes, defaultBins()), // dims, size, bins
        statsXYZ({}, stokes, defaultBins(), depth), // dims, size, bins, partial multiplier
        statsZ({height, width}, width * height * stokes) // dims, size
    {}

    Dims(hsize_t width, hsize_t height, hsize_t depth, hsize_t stokes) :
        N(4),
        width(width), height(height), depth(depth), stokes(stokes),
        standard({stokes, depth, height, width}),
        swizzled({stokes, width, height, depth}),
        mipMapExtra({stokes, depth}),
        tileDims({1, 1, TILE_SIZE, TILE_SIZE}),
        statsXY({stokes, depth}, depth * stokes, defaultBins()), // dims, size, bins
        statsXYZ({stokes}, stokes, defaultBins(), depth), // dims, size, bins, partial multiplier
        statsZ({stokes, height, width}, width * height * stokes) // dims, size
    {}
    
    int defaultBins() {
        return int(std::max(sqrt(width * height), 2.0));
    }
    
    bool useChunks() {
        return TILE_SIZE <= width && TILE_SIZE <= height;
    }
    
    static Dims makeDims(int N, long* dims) {
        if (N == 2) {
            return Dims(dims[0], dims[1]);
        } else if (N == 3) {
            return Dims(dims[0], dims[1], dims[2]);
        } else if (N == 4) {
            return Dims(dims[0], dims[1], dims[2], dims[3]);
        }
    }

    int N;
    hsize_t width, height, depth, stokes;
    
    vector<hsize_t> standard;
    vector<hsize_t> swizzled;
    vector<hsize_t> mipMapExtra;
    vector<hsize_t> tileDims;
    
    StatsDims statsXY;
    StatsDims statsXYZ;
    StatsDims statsZ;
};

struct Stats {
    Stats() {}
    
    Stats(StatsDims dims) : 
        dims(dims),
        minVals(dims.statsSize, numeric_limits<double>::max()), 
        maxVals(dims.statsSize, -numeric_limits<double>::max()),
        sums(dims.statsSize),
        sumsSq(dims.statsSize),
        nanCounts(dims.statsSize),
        histograms(dims.histSize),
        partialHistograms(dims.partialHistSize)
    {}
        
    void writeDset(Group& group, string name, vector<double>& vals, FloatType file_dtype, DataSpace dspace) {
        auto dset = group.createDataSet(name, file_dtype, dspace);
        dset.write(vals.data(), PredType::NATIVE_DOUBLE);
    }
    
    void writeDset(Group& group, string name, vector<int64_t>& vals, IntType file_dtype, DataSpace dspace) {
        auto dset = group.createDataSet(name, file_dtype, dspace);
        dset.write(vals.data(), PredType::NATIVE_INT64);
    }

    void write(Group& group, FloatType floatType, IntType intType) {
        auto statsSpace = DataSpace(dims.statsDims.size(), dims.statsDims.data());
        writeDset(group, "MIN", minVals, floatType, statsSpace);
        writeDset(group, "MAX", maxVals, floatType, statsSpace);
        writeDset(group, "SUM", sums, floatType, statsSpace);
        writeDset(group, "SUM_SQ", sumsSq, floatType, statsSpace);
        writeDset(group, "NAN_COUNT", nanCounts, intType, statsSpace);
        
        if (dims.histSize) {
            auto histSpace = DataSpace(dims.histDims.size(), dims.histDims.data());
            writeDset(group, "HISTOGRAM", histograms, intType, histSpace);
        }
    }
    
    StatsDims dims;

    vector<double> minVals;
    vector<double> maxVals;
    vector<double> sums;
    vector<double> sumsSq;
    vector<int64_t> nanCounts;
    vector<int64_t> histograms;
    vector<int64_t> partialHistograms;
};

struct MipMap {
    MipMap() {}
    
    MipMap(int N, hsize_t width, hsize_t height, hsize_t depth, int divisor) :
        N(N),
        divisor(divisor),
        channelSize(width * height),
        width(width),
        height(height),
        depth(depth),
        vals(depth * channelSize),
        count(depth * channelSize)
    {}
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t totalChannelOffset) {
        hsize_t mipIndex = totalChannelOffset * channelSize + (y / divisor) * width + (x / divisor);
        vals[mipIndex] += val;
        count[mipIndex]++;
    }
    
    void calculate() {
        for (int mipIndex = 0; mipIndex < vals.size(); mipIndex++) {
            if (count[mipIndex]) {
                vals[mipIndex] /= count[mipIndex];
            } else {
                vals[mipIndex] = NAN;
            }
        }
    }
    
    void reset() {
        std::fill(vals.begin(), vals.end(), 0);
        std::fill(count.begin(), count.end(), 0);
    }
    
    bool useChunks() {
        return TILE_SIZE <= width && TILE_SIZE <= height;
    }
    
    void createDataset(Group mipMapGroup, FloatType floatType, Dims dims) {
        vector<hsize_t> mipMapDims = dims.mipMapExtra;
        mipMapDims.push_back(height);
        mipMapDims.push_back(width);
        
        DSetCreatPropList createPlist;
        if (useChunks()) {
            createPlist.setChunk(dims.N, dims.tileDims.data());
        }
        
        auto dataSpace = DataSpace(N, mipMapDims.data());
        
        ostringstream mipMapName;
        mipMapName << "DATA_XY_" << divisor;
        
        dataSet = mipMapGroup.createDataSet(mipMapName.str().c_str(), floatType, dataSpace, createPlist);
    }
    
    void write(hsize_t stokesOffset, hsize_t channelOffset) {
        vector<hsize_t> count;
        vector<hsize_t> start;
        
        if (N == 2) {
            count = {height, width};
            start = {0, 0};
        } else if (N == 3) {
            count = {depth, height, width};
            start = {channelOffset, 0, 0};
        } else if (N == 4) {
            count = {1, depth, height, width};
            start = {stokesOffset, channelOffset, 0, 0};
        }
        
        hsize_t memDims[] = {depth, height, width};
        DataSpace memspace(3, memDims);

        auto sliceDataSpace = dataSet.getSpace();
        sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

        dataSet.write(vals.data(), PredType::NATIVE_DOUBLE, memspace, sliceDataSpace);
    }
    
    static void initialise(vector<MipMap>& mipMaps, int N, hsize_t width, hsize_t height, hsize_t depth) {
        int divisor = 1;
        while (width > MIN_MIPMAP_SIZE && height > MIN_MIPMAP_SIZE) {
            divisor *= 2;
            width = (width + 1) / 2;
            height = (height + 1) / 2;
            mipMaps.push_back(MipMap(N, width, height, depth, divisor));
        }
    }
    
    int N;
    int divisor;
    hsize_t channelSize;
    hsize_t width;
    hsize_t height;
    hsize_t depth;
    
    vector<double> vals;
    vector<hsize_t> count;
    
    DataSet dataSet;
};

struct TimerCounter {
    TimerCounter() : value(0) {}
    
    TimerCounter(unsigned int value) : value(value) {}
    
    TimerCounter operator+(const TimerCounter& other) {
        return TimerCounter(this->value + other.value);
    }
    
    void start() {
        startTime = chrono::high_resolution_clock::now();
    }
    
    void stop() {
        auto stopTime = chrono::high_resolution_clock::now();
        value += chrono::duration_cast<chrono::milliseconds>(stopTime - startTime).count();
    }
    
    double seconds() {
        return value * 1e-3;
    }
    
    double speed(hsize_t imageSize) {
        return (imageSize * 4) * 1.0e-6 / seconds();
    }
    
    unsigned int value;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

struct Timer {
    Timer() {}
    Timer(hsize_t imageSize, bool slow) : 
        size(imageSize),
        loopLabel1(slow ? "XY, XYZ & Z stats" : "Swizzling, XY & XYZ stats"),
        loopLabel2(slow ? "Histograms" : "Z stats"),
        loopLabel3(slow ? "Swizzling" : "Histograms"),
        loopLabel4(slow ? "????" : "MipMaps")
    {}
    
    hsize_t size;
    
    TimerCounter alloc;
    TimerCounter read;
    TimerCounter process1;
    TimerCounter process2;
    TimerCounter process3;
    TimerCounter process4;
    TimerCounter write;
    
    string loopLabel1;
    string loopLabel2;
    string loopLabel3;
    string loopLabel4;
    
    void print() {
        TimerCounter process = process1 + process2 + process3 + process4;
        TimerCounter total = alloc + read + process + write;
        
        cout << endl;
        cout << "Allocated in " << alloc.seconds() << " seconds (" << alloc.speed(size) << " MB/s)" << endl;
        cout << "Read in " << read.seconds() << " seconds (" << read.speed(size) << " MB/s)" << endl;
        cout << "Processed in " << process.seconds() << " seconds (" << process.speed(size) << " MB/s)" << endl;
        cout << "\t" << loopLabel1 << ": " << process1.seconds() << endl;
        cout << "\t" << loopLabel2 << ": " << process2.seconds() << endl;
        cout << "\t" << loopLabel3 << ": " << process3.seconds() << endl;
        cout << "\t" << loopLabel4 << ": " << process4.seconds() << endl;
        cout << "Written in " << write.seconds() << " seconds (" << write.speed(size) << " MB/s)" << endl;
        cout << "TOTAL: " << total.seconds() << " seconds (" << total.speed(size) << " MB/s)" << endl;
    }
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
        status(0),
        strType(PredType::C_S1, 256),
        boolType(PredType::NATIVE_HBOOL), 
        doubleType(PredType::NATIVE_DOUBLE),
        floatType(PredType::NATIVE_FLOAT),
        intType(PredType::NATIVE_INT64)
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
    
    ~Image() {
        outputFile.close();
    }
    
    void createOutputFile() {
        outputFile = H5File(tempOutputFileName, H5F_ACC_TRUNC);
        outputGroup = outputFile.createGroup("0");
        
        DSetCreatPropList standardCreatePlist;
        if (dims.useChunks()) {
            standardCreatePlist.setChunk(N, dims.tileDims.data());
        }
        auto standardDataSpace = DataSpace(N, dims.standard.data());
        standardDataSet = outputGroup.createDataSet("DATA", floatType, standardDataSpace, standardCreatePlist);

        if (depth > 1) {
            auto swizzledGroup = outputGroup.createGroup("SwizzledData");
            auto swizzledDataSpace = DataSpace(N, dims.swizzled.data());
            swizzledDataSet = swizzledGroup.createDataSet(swizzledName, floatType, swizzledDataSpace);
        }
        
        // I don't know if this naming convention still makes sense, but I'm replicating the schema for now
        auto mipMapGroup = outputGroup.createGroup("MipMaps").createGroup("DATA");
        
        for (auto& mipMap : mipMaps) {
            mipMap.createDataset(mipMapGroup, floatType, dims);
        }
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
        
                        if (status != 0) {
                            throw "Could not read string attribute";
                        }
                        
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
    
    void slowSwizzle() {
        hsize_t sliceSize;
        
        if (N == 3) {
            sliceSize = depth * TILE_SIZE * TILE_SIZE;
        } else if (N == 4) {
            sliceSize = stokes * depth * TILE_SIZE * TILE_SIZE;
        }
        
        cout << "Allocating " << sliceSize * 4 * 2 * 1e-9 << " GB of memory... " << endl;
        timer.alloc.start();
        
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];

        timer.alloc.stop();
        
        auto standardDataSpace = standardDataSet.getSpace();
        auto swizzledDataSpace = swizzledDataSet.getSpace();
        
        for (unsigned int s = 0; s < stokes; s++) {
            for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                    hsize_t xSize = min(TILE_SIZE, width - xOffset);
                    hsize_t ySize = min(TILE_SIZE, height - yOffset);
                    
                    vector<hsize_t> standardMemDims;
                    vector<hsize_t> swizzledMemDims;
                    vector<hsize_t> standardOffset;
                    vector<hsize_t> standardCount;
                    vector<hsize_t> swizzledOffset;
                    vector<hsize_t> swizzledCount;
                    
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
                    
                    DataSpace standardMemspace(standardMemDims.size(), standardMemDims.data());
                    DataSpace swizzledMemspace(swizzledMemDims.size(), swizzledMemDims.data());
                    
                    // read tile slice
                    timer.read.start();
                    
                    standardDataSpace.selectHyperslab(H5S_SELECT_SET, standardCount.data(), standardOffset.data());
                    standardDataSet.read(standardSlice, PredType::NATIVE_FLOAT, standardMemspace, standardDataSpace);
                    
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
                    swizzledDataSet.write(rotatedSlice, PredType::NATIVE_FLOAT, swizzledMemspace, swizzledDataSpace);
                    
                    timer.write.stop();
                }
            }
        }
        
        delete[] standardSlice;
        delete[] rotatedSlice;
    }
    
    void allocate(hsize_t cubeSize) {
        cout << "Allocating " << cubeSize * 4 * 2 * 1e-9 << " GB of memory... " << endl;
        timer.alloc.start();

        standardCube = new float[cubeSize];
        
        statsXY = Stats(dims.statsXY);
        
        if (depth > 1) {
            statsZ = Stats(dims.statsZ);
            statsXYZ = Stats(dims.statsXYZ);
        }

        timer.alloc.stop();
    }
    
    void allocateSwizzled(hsize_t rotatedSize) {
        if (depth > 1) {
            timer.alloc.start();
            rotatedCube = new float[rotatedSize];
            timer.alloc.stop();
        }
    }
    
    void freeSwizzled() {
        if (depth > 1) {
            delete[] rotatedCube;
        }
    }
    
    void readFits(long* fpixel, int cubeSize) {
        timer.read.start();
        fits_read_pix(inputFilePtr, TFLOAT, fpixel, cubeSize, NULL, standardCube, NULL, &status);
        timer.read.stop();
        
        if (status != 0) {
            throw "Could not read image data";
        }
    }
    
    void fastCopy() {
        auto cubeSize = depth * height * width;
        allocate(cubeSize);

        for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
            // Read data into memory space
            
            long fpixel[] = {1, 1, 1, currentStokes + 1};
            cout << "Reading Stokes " << currentStokes << " dataset... " << endl;
            readFits(fpixel, cubeSize);
            
            // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
            allocateSwizzled(cubeSize);

            cout << "Processing Stokes " << currentStokes << " dataset..." << endl;
            timer.process1.start();
            
            cout << " * XY statistics" << (depth > 1 ? " and fast swizzling" : "") <<  "... " << endl;

            // First loop calculates stats for each XY slice and rotates the dataset
#pragma omp parallel for
            for (auto i = 0; i < depth; i++) {
                float minVal = numeric_limits<float>::max();
                float maxVal = -numeric_limits<float>::max();
                double sum = 0;
                double sumSq = 0;
                int64_t nanCount = 0;

                for (auto j = 0; j < height; j++) {
                    for (auto k = 0; k < width; k++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto destIndex = i + depth * j + (height * depth) * k;
                        auto val = standardCube[sourceIndex];
                        
                        if (depth > 1) {
                            rotatedCube[destIndex] = val;
                        }
                        
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
                cout << " * XYZ statistics... " << endl;
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
            
            timer.process1.stop();

            if (depth > 1) {
                cout << " * Z statistics... " << endl;
                // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)
                
                timer.process2.start();
#pragma omp parallel for
                for (auto j = 0; j < height; j++) {
                    for (auto k = 0; k < width; k++) {
                        float minVal = numeric_limits<float>::max();
                        float maxVal = -numeric_limits<float>::max();
                        double sum = 0;
                        double sumSq = 0;
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
                        
                        if (nanCount != depth) {
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
                
                timer.process2.stop();
            }
            
            cout << " * Histograms... " << endl;

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

            timer.process3.stop();

            cout << "Writing Stokes " << currentStokes << " dataset... " << endl;
            
            timer.write.start();
                        
            vector<hsize_t> count;
            vector<hsize_t> start;
            
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
            DataSpace memspace(3, memDims);

            auto sliceDataSpace = standardDataSet.getSpace();
            sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

            standardDataSet.write(standardCube, PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
            
            if (depth > 1) {
                // This all technically worked if we reused the standard filespace and memspace
                // But it's probably not a good idea to rely on two incorrect values cancelling each other out
                vector<hsize_t> swizzledCount;
                
                if (N == 3) {
                    swizzledCount = {width, height, depth};
                } else if (N == 4) {
                    swizzledCount = {1, width, height, depth};
                }
                
                hsize_t swizzledMemDims[] = {width, height, depth};
                DataSpace swizzledMemspace(3, swizzledMemDims);
                
                auto swizzledDataSpace = swizzledDataSet.getSpace();
                swizzledDataSpace.selectHyperslab(H5S_SELECT_SET, swizzledCount.data(), start.data());
                
                swizzledDataSet.write(rotatedCube, PredType::NATIVE_FLOAT, swizzledMemspace, swizzledDataSpace);
            }

            timer.write.stop();
            
            // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
            
            freeSwizzled();
            
            // Fourth loop handles mipmaps
            
            // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
            
            cout << " * MipMaps... " << endl;

            timer.process4.start();
            
#pragma omp parallel for
            for (auto c = 0; c < depth; c++) {
                for (auto y = 0; y < height; y++) {
                    for (auto x = 0; x < width; x++) {
                        auto sourceIndex = x + width * y + (height * width) * c;
                        auto val = standardCube[sourceIndex];
                        if (!isnan(val)) {
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
    
    void slowCopy() {
        // Allocate one channel at a time, and no swizzled data
        auto cubeSize = height * width;
        allocate(cubeSize);
        
        auto sliceDataSpace = standardDataSet.getSpace();
                            
        vector<hsize_t> count;
        vector<hsize_t> start;
        
        if (N == 2) {
            count = {height, width};
        } else if (N == 3) {
            count = {1, height, width};
        } else if (N == 4) {
            count = {1, 1, height, width};
        }
        
        hsize_t memDims[] = {height, width};
        DataSpace memspace(2, memDims);
        
        for (unsigned int s = 0; s < stokes; s++) {
            cout << "Processing Stokes " << s << " dataset... " << endl;
            
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
                standardDataSet.write(standardCube, PredType::NATIVE_FLOAT, memspace, sliceDataSpace);
                timer.write.stop();
                
                timer.process1.start();
                
                auto indexXY = s * depth + c;
                
                for (auto y = 0; y < height; y++) {
                    for (auto x = 0; x < width; x++) {
                        auto pos = y * width + x; // relative to channel slice
                        auto indexZ = s * width * height + pos; // relative to whole image
                        
                        auto val = standardCube[pos];
                                            
                        if (!isnan(val)) {
                            // XY statistics
                            // TODO: check if indexing stats arrays inside loop is slow or is optimised away
                            
                            // This should be safe. It would only fail if we had a strictly descending or ascending sequence;
                            // very unlikely when we're iterating over all values in a channel.
                            if (val < statsXY.minVals[indexXY]) {
                                statsXY.minVals[indexXY] = val;
                            } else if (val > statsXY.maxVals[indexXY]) {
                                statsXY.maxVals[indexXY] = val;
                            }
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
                    statsXY.sums[indexXY] = NAN;
                    statsXY.sumsSq[indexXY] = NAN;
                }
                
                // Accumulate XYZ statistics
                
                if (depth > 1) {
                    if (!isnan(statsXY.sums[indexXY])) {
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
                        statsZ.sums[indexZ] = NAN;
                        statsZ.sumsSq[indexZ] = NAN;
                    }
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
                cubeHist = !isnan(cubeMin) && !isnan(cubeMax) && cubeRange > 0;
            }
            
            for (hsize_t c = depth; c-- > 0; ) {                
                auto indexXY = s * depth + c;
                                
                double chanMin = statsXY.minVals[indexXY];
                double chanMax = statsXY.maxVals[indexXY];
                double chanRange = chanMax - chanMin;
                bool chanHist(!isnan(chanMin) && !isnan(chanMax) && chanRange > 0);
                
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
                        if (!isnan(val)) {
                            if (chanHist) {
                                int binIndex = min(numBinsXY - 1, (int)(numBinsXY * (val - chanMin) / chanRange));
                                statsXY.histograms[s * depth * numBinsXY + c * numBinsXY + binIndex]++;
                            }
                            
                            if (cubeHist) {
                                int binIndex = min(numBinsXYZ - 1, (int)(numBinsXYZ * (val - cubeMin) / cubeRange));
                                statsXYZ.histograms[s * numBinsXYZ + binIndex]++;
                            }
                        }
                } // end of XY loop
            } // end of second channel loop (XY and XYZ histograms)
            
            timer.process2.stop();
            
        } // end of stokes
                
        // Swizzle
        if (depth > 1) {
            cout << "Performing slow, memory-saving rotation." << endl;
            slowSwizzle();
        }
    }
    
    void convert() {
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
    
    
private:
    string tempOutputFileName;
    string outputFileName;
    fitsfile* inputFilePtr;
    
    // Main HDF5 objects
    H5File outputFile;
    Group outputGroup;
    DataSet standardDataSet;
    DataSet swizzledDataSet;
    
    // Data objects
    float* standardCube;
    float* rotatedCube;
    
    // Stats
    Stats statsXY;
    Stats statsZ;
    Stats statsXYZ;
    
    // MipMaps
    vector<MipMap> mipMaps;
    
    int status;
    bool slow;
    Timer timer;
    
    int N;
    hsize_t stokes, depth, height, width;
    Dims dims;
    int numBinsXY;
    int numBinsXYZ;
    string swizzledName;
    
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
    bool slow(false);
    
    if (!getOptions(argc, argv, inputFileName, outputFileName, slow)) {
        return 1;
    }
    
    Image image;
        
    try {
        image = Image(inputFileName, outputFileName, slow);
    
        cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << endl;

        image.convert();
    } catch (const char* msg) {
        cerr << "Error: " << msg << ". Aborting." << endl;
        return 1;
    }

    return 0;
}
