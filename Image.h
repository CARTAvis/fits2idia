#ifndef __IMAGE_H
#define __IMAGE_H

#include "common.h"
#include "Dims.h"
#include "Stats.h"
#include "MipMap.h"
#include "Timer.h"

class Image {
public:
    Image() {}
    Image(std::string inputFileName, std::string outputFileName, bool slow);
    ~Image();
    
    void createOutputFile();
    void copyHeaders();
    void slowSwizzle();
    void allocate(hsize_t cubeSize);
    void allocateSwizzled(hsize_t rotatedSize);
    void freeSwizzled();
    void readFits(long* fpixel, int cubeSize);
    
    void fastCopy();
    void slowCopy();
    void convert();
    
private:
    std::string tempOutputFileName;
    std::string outputFileName;
    fitsfile* inputFilePtr;
    
    // Main HDF5 objects
    H5::H5File outputFile;
    H5::Group outputGroup;
    H5::DataSet standardDataSet;
    H5::DataSet swizzledDataSet;
    
    // Data objects
    float* standardCube;
    float* rotatedCube;
    
    // Stats
    Stats statsXY;
    Stats statsZ;
    Stats statsXYZ;
    
    // MipMaps
    std::vector<MipMap> mipMaps;
    
    int status;
    bool slow;
    Timer timer;
    
    int N;
    hsize_t stokes, depth, height, width;
    Dims dims;
    int numBinsXY;
    int numBinsXYZ;
    std::string swizzledName;
    
    // Types
    H5::StrType strType;
    H5::IntType boolType;
    H5::FloatType doubleType;
    H5::FloatType floatType;
    H5::IntType intType;
};

#endif
