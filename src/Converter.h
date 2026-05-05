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
    Converter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    virtual ~Converter();
    
    static std::unique_ptr<Converter> getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool progress, bool zMips);
    void convert();
    void reportMemoryUsage();
    virtual MemoryUsage calculateMemoryUsage() = 0;

    // checks the order of STOKES and FREQUENCY axis an if it is STOKES,FREQ sets flat swapStokesFreqAxis to true:
    bool checkIfSwapAxisRequired();
    
    // reduce memory usage:
    bool ReduceMemoryUsage( hsize_t memoryLimit, int max_iter=10 );
    
    void SetChunkDivider( int divider );
    
protected:
    virtual void copyAndCalculate() = 0;
    
    Timer timer;
    bool progress;
    bool zMips;
    
    std::string tempOutputFileName;
    std::string outputFileName;
    fitsfile* inputFilePtr;
    bool      swapStokesFreqAxis;
    
    // Main HDF5 objects
    H5::H5File outputFile;
    H5::Group outputGroup;
    H5::DataSet standardDataSet;
    H5::DataSet swizzledDataSet;
    
    float* standardCube;
    float* rotatedCube;
    
    // Stats
    Stats statsXY;  // per channel histogram
    Stats statsZ;
    Stats statsXYZ; // per cube histogram (using entire cube)
    
    // MipMaps
    MipMaps mipMaps;
    
    int N;
    hsize_t stokes, depth, height, width;
    hsize_t numBins;
    
    // optimisations :
    hsize_t height_chunk; // block in height used for partial histograms calculations = height / height_divider
    int height_divider; // this specify how to divide height for partial histograms OpenMP optimisation 
                        // this is required when too much memory is required without any division
    
    // Dataset dimensions    
    std::vector<hsize_t> standardDims;
    std::vector<hsize_t> swizzledDims;
    std::vector<hsize_t> tileDims;
    
    std::string swizzledName;
};


class FastConverter : public Converter {
public:
    FastConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};


class SlowConverter : public Converter {
public:
    SlowConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};

#endif
