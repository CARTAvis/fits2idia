#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"

struct Stats {
    Stats() {}
    Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins = 0);
    
    static hsize_t size(std::vector<hsize_t> dims, hsize_t numBins = 0, hsize_t partialHistMultiplier = 0);
    
    // Setup
    void createDatasets(H5::Group group, std::string name);
    void createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier = 0);
    
    // Basic stats

    void accumulateFinite(hsize_t index, float val) {
        minVals[index] = fmin(minVals[index], val);
        maxVals[index] = fmax(maxVals[index], val);
        sums[index] += val;
        sumsSq[index] += val * val;
    }

    void accumulateFiniteLazy(hsize_t index, float val) {
        if (val < minVals[index]) {
            minVals[index] = val;
        } else if (val > maxVals[index]) {
            maxVals[index] = val;
        }
        sums[index] += val;
        sumsSq[index] += val * val;
    }

    void accumulateFiniteLazyFirst(hsize_t index, float val) {
        minVals[index] = val;
        maxVals[index] = val;
        sums[index] += val;
        sumsSq[index] += val * val;
    }

    void accumulateNonFinite(hsize_t index) {
        nanCounts[index]++;
    }

    void accumulateStats(const Stats& other, hsize_t index, hsize_t otherIndex) {
        if (std::isfinite(other.maxVals[otherIndex])) {
            sums[index] += other.sums[otherIndex];
            sumsSq[index] += other.sumsSq[otherIndex];
            minVals[index] = fmin(minVals[index], other.minVals[otherIndex]);
            maxVals[index] = fmax(maxVals[index], other.maxVals[otherIndex]);
        }
        nanCounts[index] += other.nanCounts[otherIndex];
    }

    void finalMinMax(hsize_t index, hsize_t totalVals) {
        if ((hsize_t)nanCounts[index] == totalVals) {
            minVals[index] = NAN;
            maxVals[index] = NAN;
        }
    }
   
    // Histograms

    void accumulateHistogram(float val, double min, double range, hsize_t offset) {
        int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
        histograms[offset * numBins + binIndex]++;
    }

    void accumulatePartialHistogram(float val, double min, double range, hsize_t offset) {
        int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
        partialHistograms[offset * numBins + binIndex]++;
    }

    void consolidatePartialHistogram() {
        auto multiplier = partialHistograms.size() / histograms.size();
        for (hsize_t offset = 0; offset < multiplier; offset++) {
            for (hsize_t binIndex = 0; binIndex < numBins; binIndex++) {
                histograms[binIndex] += partialHistograms[offset * numBins + binIndex];
            }
        }
    }
    
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
