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
    void accumulateStats(Stats stats, hsize_t index);

    float minVal;
    float maxVal;
    double sum;
    double sumSq;
    int64_t nanCount;
};

struct Stats {
    Stats() {}
    Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins = 0);
    
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
    
    void resetBuffers();
    
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

    // Buffers
    std::vector<float> minVals;
    std::vector<float> maxVals;
    std::vector<double> sums;
    std::vector<double> sumsSq;
    std::vector<int64_t> nanCounts;
    
    std::vector<int64_t> histograms;
    std::vector<int64_t> partialHistograms;
};

#endif
