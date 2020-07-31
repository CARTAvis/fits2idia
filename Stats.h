#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"
#include "HDF5Wrapper.h"

struct StatsCounter {
    StatsCounter() : minVal(std::numeric_limits<float>::max()), maxVal(-std::numeric_limits<float>::max()), sum(0), sumSq(0), nanCount(0) {
    }

    void accumulateFinite(float val) {
        minVal = fmin(minVal, val);
        maxVal = fmax(maxVal, val);
        sum += val;
        sumSq += val * val;
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
    //void createDatasets(H5::Group group, std::string name);
    void createDatasets(H5OutputFile &H5outputfile, hid_t gid, std::string name);
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

    // Writing
    void write(H5OutputFile &H5outputfile);
    void write(H5OutputFile &H5outputfile, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start);
    void write(H5OutputFile &H5outputfile, const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start);
    void writeBasic(H5OutputFile &H5outputfile, const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);
    void writeHistogram(H5OutputFile &H5outputfile, const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);

    // Dataset dimensions
    std::vector<hsize_t> basicDatasetDims;
    hsize_t numBins;

    // Datasets
//     H5::DataSet minDset;
//     H5::DataSet maxDset;
//     H5::DataSet sumDset;
//     H5::DataSet ssqDset;
//     H5::DataSet nanDset;
//
//     H5::DataSet histDset;

    hid_t minDset;
    hid_t maxDset;
    hid_t sumDset;
    hid_t ssqDset;
    hid_t nanDset;

    hid_t histDset;


    // Buffer dimensions

    std::vector<hsize_t> fullBasicBufferDims;
    hsize_t partialHistMultiplier;

    // Buffers
    //why not change this to vectors ???
    float* minVals;
    float* maxVals;
    double* sums;
    double* sumsSq;
    int64_t* nanCounts;

    int64_t* histograms;
    int64_t* partialHistograms;
};

#endif
