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

enum eSmartConverterType { eAutoSelectedSmartConverter=0, eSmartConverterSpatialParallel=1, eSmartConverterChannelParallel=2 };

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
    
    static std::unique_ptr<Converter> getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool smart, eSmartConverterType smarttype, bool progress, bool zMips, int memoryLimitInMb, bool auto_mode);
    void convert();
    void reportMemoryUsage();
    virtual MemoryUsage calculateMemoryUsage() = 0;

    // checks the order of STOKES and FREQUENCY axis an if it is STOKES,FREQ sets flat swapStokesFreqAxis to true:
    bool checkIfSwapAxisRequired();
    
    // reduce memory usage:
    bool ReduceMemoryUsage( hsize_t memoryLimit, int max_iter=10 );
    
    void SetChunkDivider( int divider );
    
    void setIOBlocks( int _n_io_blocks ){ n_io_blocks = _n_io_blocks; }
    
    void getDimensions( hsize_t& _stokes, hsize_t& _depth, hsize_t& _height, hsize_t& _width );
    
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
    Stats statsXY;  // per channel stats and histogram
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
    int n_io_blocks; // number of channel images read at once to optimise I/O to read larger portions of file
    
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

class SmartConverter : public Converter {
public:
    SmartConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
//    SmartConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMip);
    MemoryUsage calculateMemoryUsage() override;
    void setMemoryLimit(int _memoryLimitInMb){ memoryLimitInMb = _memoryLimitInMb; }

    // flag to enable single pass through the data 
    // XYZ (cube) histogram is calculated using channel histograms 
    // which means it's approximate only, but this is "good enough" for the visualisation purposes
    static bool bApproximateCubeHistogram;

protected:
    int memoryLimitInMb;
 
    void copyAndCalculate() override;
    
    // optional second pass to calculate exect XYZ (cube) histogram:
    // double doSecondPass( unsigned int s, double& total_io_ms );
    
    // SmartFastConverter.cc requires extra arguments:
    virtual double doSecondPass( unsigned int s, int n_blocks, int sliceIncrement, int leftOverSlices, double& total_io_ms );
    
    // calculates approximate XYZ (cube) histogram using channel histograms
    // it is not exact, but good enough for visualisation purposes
    double calcApproxCubeHistogram( unsigned int s );
    
    // calculate Stats for channel (min/max etc):
    // Parameters:
    //     indexXY - channel (c)
    //     block_pos - position of the block to be processed
    //
    // Return value:
    //     execution time in milli-seconds
    // TODO : check if I can simplify these parameters a bit more 
    double calculateChannelStats( hsize_t indexXY, hsize_t block_pos );

    // calculate channel histograms
    // Parameters:
    //     indexXY - channel (c)
    //     block_pos - position of the block to be processed
    //
    // Return value:
    //     execution time in milli-seconds
    // TODO : check if I can simplify these parameters a bit more     
    double calculateChannelHistogram( hsize_t indexXY, hsize_t block_pos );
    
    // calculate rotated dataset:
    // Return value:
    //     execution time in milli-seconds
    // TODO : check if I can simplify these parameters a bit more     
    double calculateRotatedData(double& total_io_ms);
    
    // calculate rotated channel:
    // Return value:
    //     execution time in milli-seconds
    // TODO : check if I can simplify these parameters a bit more     
    double calculateRotatedChannel( unsigned int s, hsize_t c_start, hsize_t c_end, float* standardCube, float* rotatedCube, int sliceIncrement );
    
    void ReadRotateWriteFullDepth(float* standardSliceToRead, float* rotatedSliceToWrite, unsigned int s, hsize_t xStart, hsize_t yStart,
        hsize_t xLimit, hsize_t yLimit, hsize_t xIncrement, hsize_t yIncrement);
    void ReadRotateWritePartialDepth(float* standardSliceToRead, float* rotatedSliceToWrite, unsigned int s, hsize_t xStart, hsize_t yStart,
        hsize_t zStart, hsize_t xLimit, hsize_t yLimit, hsize_t zLimit, hsize_t xIncrement, hsize_t yIncrement, hsize_t zIncrement);
};

class SmartFastConverter : public SmartConverter {
public:
    SmartFastConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;

    // SmartFastConverter.cc requires extra arguments:
    virtual double doSecondPass( unsigned int s, int n_blocks, int sliceIncrement, int leftOverSlices, double& total_io_ms ) override;
};    



class SlowConverter : public Converter {
public:
    SlowConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
protected:
    void copyAndCalculate() override;
};

#endif
