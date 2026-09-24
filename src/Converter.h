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
#include "IOCost.h"

enum eSmartConverterType { eAutoSelectedSmartConverter=0, eSmartConverterSpatialParallel=1, eSmartConverterChannelParallel=2, eSmartMicroMemoryConverter=3 };

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
    
    
    // static factory function to get the required or automatically suggested converter object:
    static std::unique_ptr<Converter> getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool smart, eSmartConverterType smarttype, bool progress, bool zMips, int memoryLimitInMb, bool auto_mode);
    static std::unique_ptr<Converter> getOptimalConverter(std::string inputFileName, std::string outputFileName, bool slow, bool smart, eSmartConverterType smarttype, bool progress, bool zMips, int memoryLimitInMb, bool auto_mode);    
    
    void convert();
    void reportMemoryUsage();
    void reportExecTime();
    void reportMemoryAndExecTime();
    virtual MemoryUsage calculateMemoryUsage() = 0;
    
    // parse list of included and excluded datasets to be saved in the output file(s):
    void ParseExcludeIncludeOptions( std::string exclude_list, std::string include_list );

    // checks the order of STOKES and FREQUENCY axis an if it is STOKES,FREQ sets flat swapStokesFreqAxis to true:
    bool checkIfSwapAxisRequired();
    
    // reduce memory usage:
    virtual bool ReduceMemoryUsage( hsize_t memoryLimit, int max_iter=10 );
    
    // calculate IO cost :
    virtual IOCostBreakdown estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel);
    
    void setIOBlocks( int _n_io_blocks ){ n_io_blocks = _n_io_blocks; }
    
    void setSystemName(const char* system_name );
    
    void getDimensions( hsize_t& _stokes, hsize_t& _depth, hsize_t& _height, hsize_t& _width );
    
    // functions about the type of the converter and its parameters to be used 
    // in saving metadata to the output HDF5 file:
    virtual const char* getConverterType() = 0;
    
    bool getCubeHistogramApproximated(){ return false; }
    
    // options:
    static bool rotatedDatasetChunking;
    
protected:
    virtual void copyAndCalculate() = 0;
    
    void DebugDimsAndParameters( const std::vector<hsize_t>& swizzledDims, const std::vector<hsize_t>& swizzledChunkDims, const H5::DataSet& swizzledDataSet );
    
    Timer timer;
    bool progress;
    bool zMips;
    std::string systemName; // can help with predictions by using system-specific measured BWs etc
    
    std::string tempOutputFileName;
    std::string outputFileName;
    fitsfile* inputFilePtr;
    bool      swapStokesFreqAxis;
    
    // flags which Datasets to created in the output file(s):
    // empty set means - all the datasets are created
    // when the set is not empty then only the specified datasets will be saved:
    std::unordered_map<std::string, bool> output_datasets;
    bool includeDataset(const char* dataset);
    
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
//    hsize_t height_chunk; // block in height used for partial histograms calculations = height / height_divider
//    int height_divider; // this specify how to divide height for partial histograms OpenMP optimisation 
                        // this is required when too much memory is required without any division
    int n_io_blocks; // number of channel images read at once to optimise I/O to read larger portions of file
//    int min_mipmap_threads; // minimum number of MipMap threads in OMP version (otherwise = CONST = 1)
    
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
    
    virtual const char* getConverterType() override { return "FAST"; }
    
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
    
    // functions about the type of the converter and its parameters to be used 
    // in saving metadata to the output HDF5 file:
    virtual const char* getConverterType() override { return "SMART-XY-PARALLEL"; }
    
    bool getCubeHistogramApproximated(){  return bApproximateCubeHistogram; }
    
    // reduce memory usage:
    // needs to be public to be used in a static function in Converter
    virtual bool ReduceMemoryUsage( hsize_t memoryLimit, int max_iter=10 );
    
    // calculate IO cost :
    virtual IOCostBreakdown estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel);

protected:
    int memoryLimitInMb;
    int allowed_mipmaps_threads;
 
    void copyAndCalculate() override;
    
    // calculate rotated dataset:
    // Return value:
    //     execution time in milli-seconds
    // used to be double calculateRotatedData(double& total_io_ms)
    virtual double calculateRotatedDataAndCubeHistogram(double& total_io_ms,
                                                        const std::vector<double>& savedCubeMin,
                                                        const std::vector<double>& savedCubeMax);
    
    // optional second pass to calculate exect XYZ (cube) histogram:
    // double doSecondPass( unsigned int s, double& total_io_ms );
    
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
    
    // calculate rotated channel:
    // Return value:
    //     execution time in milli-seconds
    // TODO : check if I can simplify these parameters a bit more     
    double calculateRotatedChannel( unsigned int s, hsize_t c_start, hsize_t c_end, float* standardCube, float* rotatedCube, int sliceIncrement );
    
    void SetChunkDivider( int divider );
    
    
    void ReadRotateWriteFullDepth(float* standardSliceToRead, float* rotatedSliceToWrite, unsigned int s, hsize_t xStart, hsize_t yStart,
        hsize_t xLimit, hsize_t yLimit, hsize_t xIncrement, hsize_t yIncrement);
    void ReadRotateWritePartialDepth(float* standardSliceToRead, float* rotatedSliceToWrite, unsigned int s, hsize_t xStart, hsize_t yStart,
        hsize_t zStart, hsize_t xLimit, hsize_t yLimit, hsize_t zLimit, hsize_t xIncrement, hsize_t yIncrement, hsize_t zIncrement);
         
    // optimisations :
    hsize_t height_chunk; // block in height used for partial histograms calculations = height / height_divider
    int height_divider; // this specify how to divide height for partial histograms OpenMP optimisation 
                        // this is required when too much memory is required without any division
    int min_mipmap_threads; // minimum number of MipMap threads in OMP version (otherwise = CONST = 1)       
    
    // auxiliary objects used in calculations:
    std::vector<MipMaps> thread_mipmaps_array;
};

class SmartFastConverter : public SmartConverter {
public:
    SmartFastConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
    // functions about the type of the converter and its parameters to be used 
    // in saving metadata to the output HDF5 file:
    virtual const char* getConverterType() override { return "SMART-CHAN-PARALLEL"; }
    
    virtual IOCostBreakdown estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel) override;
    
protected:
    void copyAndCalculate() override;

    // SmartFastConverter.cc requires extra arguments:
    virtual double doSecondPass( unsigned int s, int n_blocks, int sliceIncrement, int leftOverSlices, double& total_io_ms );
    
    // calculates approximate XYZ (cube) histogram using channel histograms
    // it is not exact, but good enough for visualisation purposes
    double calcApproxCubeHistogram( unsigned int s );
};    



class SlowConverter : public Converter {
public:
    SlowConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    
    virtual const char* getConverterType() override { return "SLOW"; }
    
    virtual IOCostBreakdown estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel) override;
    
protected:
    void copyAndCalculate() override;
};

class MicroMemoryConverter : public SmartConverter {
public:
    MicroMemoryConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips);
    MemoryUsage calculateMemoryUsage() override;
    virtual bool ReduceMemoryUsage(hsize_t memoryLimit, int max_iter) override;
    virtual IOCostBreakdown estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel) override;

    virtual const char* getConverterType() override { return "MICRO-MEMORY"; }

protected:
    void copyAndCalculate() override;
};


#endif
