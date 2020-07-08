#ifndef __IMAGE_H
#define __IMAGE_H

#include "common.h"
#include "Dims.h"
#include "Stats.h"
#include "MipMap.h"
#include "Timer.h"
#include "Util.h"

class Converter {
public:
    Converter() {}
    Converter(std::string inputFileName, std::string outputFileName);
    ~Converter();
    
    static std::unique_ptr<Converter> getConverter(std::string inputFileName, std::string outputFileName, bool slow);
    void convert();
    virtual void reportMemoryUsage();
    
protected:
    virtual void copyAndCalculate();
    
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


class FastConverter : public Converter {
public:
    FastConverter(std::string inputFileName, std::string outputFileName);
    void reportMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};


class SlowConverter : public Converter {
public:
    SlowConverter(std::string inputFileName, std::string outputFileName);
    void reportMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};

#endif
