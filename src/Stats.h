/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"

struct StatsCounter {
    StatsCounter() : minVal(std::numeric_limits<float>::max()), maxVal(-std::numeric_limits<float>::max()), sum(0), sumSq(0), nanCount(0) {
    }
        
    void accumulateFinite(float val) {
        minVal = fmin(minVal, val);
        maxVal = fmax(maxVal, val);
        sum += val;
        sumSq += val * val;
    }
    
    void reset() {
        minVal = std::numeric_limits<float>::max();
        maxVal = -std::numeric_limits<float>::max();
        sum = 0;
        sumSq = 0;
        nanCount = 0;
    }

    void accumulateFiniteLazy(float val) {
        if (val < minVal) {
            minVal = val;
        } else if (val > maxVal) {
            maxVal = val;
        }
        sum += val;
        sumSq += val * val;
    }

    void accumulateFiniteLazyFirst(float val) {
        minVal = val;
        maxVal = val;
        sum += val;
        sumSq += val * val;
    }

    void accumulateNonFinite() {
        nanCount++;
    }
    
    void accumulateFromCounter(StatsCounter otherCounter) {
        minVal = fmin(minVal, otherCounter.minVal);
        maxVal = fmax(maxVal, otherCounter.maxVal);
        sum += otherCounter.sum;
        sumSq += otherCounter.sumSq;
        nanCount += otherCounter.nanCount;
    }

    float minVal;
    float maxVal;
    double sum;
    double sumSq;
    int64_t nanCount;
};

struct Stats {
    Stats();
    Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins = 0);
    ~Stats();
    
    static hsize_t size(std::vector<hsize_t> dims, hsize_t numBins = 0, hsize_t partialHistMultiplier = 0);
    
    // Setup
    void createDatasets(H5::Group group, std::string name);
    void createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier = 0);
    
    // Basic stats
    
    void accumulateStatsToCounter(StatsCounter& counter, hsize_t index) {
        if (std::isfinite(maxVals[index])) {
            counter.sum += sums[index];
            counter.sumSq += sumsSq[index];
            counter.minVal = fmin(counter.minVal, minVals[index]);
            counter.maxVal = fmax(counter.maxVal, maxVals[index]);
        }
        counter.nanCount += nanCounts[index];
    }
    
    void copyStatsFromCounter(hsize_t index, hsize_t totalVals, const StatsCounter& counter) {
        if ((hsize_t)counter.nanCount == totalVals) {
            minVals[index] = NAN;
            maxVals[index] = NAN;
        } else {
            minVals[index] = counter.minVal;
            maxVals[index] = counter.maxVal;
        }
        sums[index] = counter.sum;
        sumsSq[index] = counter.sumSq;
        nanCounts[index] = counter.nanCount;
    }
   
    // Histograms

    // Histograms
    
    void clearHistogramBuffers();
    void clearPartialHistogramBuffer();

    void accumulateHistogram(float val, double min, double range, hsize_t offset) {
        int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
        histograms[offset * numBins + binIndex]++;
    }

    void accumulatePartialHistogram(float val, double min, double range, hsize_t offset) {
        int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
        partialHistograms[offset * numBins + binIndex]++;
    }

    void consolidatePartialHistogram() {
        for (hsize_t offset = 0; offset < partialHistMultiplier; offset++) {
            for (hsize_t binIndex = 0; binIndex < numBins; binIndex++) {
                histograms[binIndex] += partialHistograms[offset * numBins + binIndex];
            }
        }
    }
    
    void consolidatePartialHistogram(hsize_t mainOffset) {
        for (hsize_t offset = 0; offset < partialHistMultiplier; offset++) {
            for (hsize_t binIndex = 0; binIndex < numBins; binIndex++) {
                histograms[mainOffset * numBins + binIndex] += partialHistograms[offset * numBins + binIndex];
                partialHistograms[offset * numBins + binIndex] = 0;     //reset partial histogram after consolidation
            }
        }
    }
    
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
    
    bool buffersAllocated;
    bool histogramBuffersAllocated;
};

#endif
