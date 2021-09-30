/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef __IMAGE_H
#define __IMAGE_H

#include "common.h"
#include "Stats.h"
#include "MipMap.h"
#include "Timer.h"
#include "Util.h"

struct MemoryUsage {
    MemoryUsage() : total(0) {}
    
    std::unordered_map<std::string, hsize_t> sizes;
    hsize_t total;
    std::string note;
};

class Converter {
public:
    Converter() {}
    Converter(std::string inputFileName, std::string outputFileName, bool progress);
    ~Converter();
    
    static std::unique_ptr<Converter> getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool progress);
    void convert();
    void reportMemoryUsage();
    virtual MemoryUsage calculateMemoryUsage();
    
protected:
    virtual void copyAndCalculate();
    
    Timer timer;
    bool progress;
    
    std::string tempOutputFileName;
    std::string outputFileName;
    fitsfile* inputFilePtr;
    
    // Main HDF5 objects
    H5::H5File outputFile;
    H5::Group outputGroup;
    H5::DataSet standardDataSet;
    H5::DataSet swizzledDataSet;
    
    float* standardCube;
    float* rotatedCube;
    
    // Stats
    Stats statsXY;
    Stats statsZ;
    Stats statsXYZ;
    
    // MipMaps
    MipMaps mipMaps;
    
    int N;
    hsize_t stokes, depth, height, width;
    hsize_t numBins;
    
    // Dataset dimensions
    
    std::vector<hsize_t> standardDims;
    std::vector<hsize_t> swizzledDims;
    std::vector<hsize_t> tileDims;
    
    std::string swizzledName;
};


class FastConverter : public Converter {
public:
    FastConverter(std::string inputFileName, std::string outputFileName, bool progress);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};


class SlowConverter : public Converter {
public:
    SlowConverter(std::string inputFileName, std::string outputFileName, bool progress);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};

#endif
