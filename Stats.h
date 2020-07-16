#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"

struct Stats;

struct StatsCounter {
    StatsCounter();
    
    void accumulateFinite(float val);
    void accumulateFiniteLazy(float val);
    void accumulateFiniteLazyFirst(float val);
    void accumulateNonFinite();
    void accumulateStats(const Stats& stats, hsize_t index);

    float minVal;
    float maxVal;
    double sum;
    double sumSq;
    int64_t nanCount;
};

struct Stats {
    Stats() {}
    Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins = 0);
    ~Stats();
    
    static hsize_t size(std::vector<hsize_t> dims, hsize_t numBins = 0, hsize_t partialHistMultiplier = 0);
    
    // Setup
    void createDatasets(H5::Group group, std::string name);
    void createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier = 0);
    
    // Basic stats
    void copyStatsFromCounter(hsize_t index, hsize_t totalVals, const StatsCounter& counter);
    
    // Histograms
    void accumulateHistogram(float val, double min, double range, hsize_t offset);
    void accumulatePartialHistogram(float val, double min, double range, hsize_t offset);
    void consolidatePartialHistogram();
    
    // Writing
    void write();
    void write(const std::vector<hsize_t>& count, const std::vector<hsize_t>& start);
    void write(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start);
    void writeBasic(const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);
    void writeHistogram(const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);
    
    // Dataset dimensions
    std::vector<hsize_t> basicDatasetDims;
    hsize_t numBins;
    
    // Datasets
    H5::DataSet minDset;
    H5::DataSet maxDset;
    H5::DataSet sumDset;
    H5::DataSet ssqDset;
    H5::DataSet nanDset;
    
    H5::DataSet histDset;
    
    // Buffer dimensions
    
    std::vector<hsize_t> fullBasicBufferDims;
    hsize_t partialHistMultiplier;

    // Buffers
    float* minVals;
    float* maxVals;
    double* sums;
    double* sumsSq;
    int64_t* nanCounts;
    
    int64_t* histograms;
    int64_t* partialHistograms;
};

#endif
