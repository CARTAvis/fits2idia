#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"

// TODO move the accumulation and histogram functionality in here and get it out of the converter code
// TODO split this and mipmaps into header and implementation

struct Stats {
    Stats() {}
    
    Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins = 0) : basicDatasetDims(basicDatasetDims), numBins(numBins) {}
    
    void createDatasets(H5::Group group, std::string name) {
        H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
        floatType.setOrder(H5T_ORDER_LE);
        
        H5::IntType intType(H5::PredType::NATIVE_INT64);
        intType.setOrder(H5T_ORDER_LE);
        
        createHdf5Dataset(minDset, group, "Statistics/" + name + "/MIN", floatType, basicDatasetDims);
        createHdf5Dataset(maxDset, group, "Statistics/" + name + "/MAX", floatType, basicDatasetDims);
        createHdf5Dataset(sumDset, group, "Statistics/" + name + "/SUM", floatType, basicDatasetDims);
        createHdf5Dataset(ssqDset, group, "Statistics/" + name + "/SUM_SQ", floatType, basicDatasetDims);
        createHdf5Dataset(nanDset, group, "Statistics/" + name + "/NAN_COUNT", intType, basicDatasetDims);
        
        if (numBins) {
            createHdf5Dataset(histDset, group, "Statistics/" + name + "/HISTOGRAM", intType, extend(basicDatasetDims, {numBins}));
        }
    }
    
    void createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier = 0) {
        fullBasicBufferDims = dims;
        auto statsSize = product(dims);
        
        minVals.resize(statsSize, std::numeric_limits<double>::max());
        maxVals.resize(statsSize, -std::numeric_limits<double>::max());
        sums.resize(statsSize);
        sumsSq.resize(statsSize);
        nanCounts.resize(statsSize);
        
        if (numBins) {
            histograms.resize(statsSize * numBins);
            partialHistograms.resize(statsSize * numBins * partialHistMultiplier);
        }
    }
    
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
    
    void accumulateStats(Stats other, hsize_t index, hsize_t otherIndex) {
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
    
    void resetBuffers() {
        std::fill(minVals.begin(), minVals.end(), std::numeric_limits<double>::max());
        std::fill(maxVals.begin(), maxVals.end(), -std::numeric_limits<double>::max());
        std::fill(sums.begin(), sums.end(), 0);
        std::fill(sumsSq.begin(), sumsSq.end(), 0);
        std::fill(nanCounts.begin(), nanCounts.end(), 0);
        
        if (numBins) {
            std::fill(histograms.begin(), histograms.end(), 0);
            std::fill(partialHistograms.begin(), partialHistograms.end(), 0);
        }
    }
    
    void write() {
        writeBasic(fullBasicBufferDims);
        
        if (numBins) {
            writeHistogram(fullBasicBufferDims);
        }
    }
    
    void write(const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
        write(fullBasicBufferDims, count, start);
    }
    
    void write(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
        auto basicN = basicDatasetDims.size();
        writeBasic(basicBufferDims, trimAxes(count, basicN), trimAxes(start, basicN));
        
        if (numBins) {
            auto histN = basicN + 1;
            writeHistogram(basicBufferDims, trimAxes(extend(count, {numBins}), histN), trimAxes(extend(start, {0}), histN));
        }
    }
        
    void writeBasic(const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS) {
        writeHdf5Data(minDset, minVals.data(), basicBufferDims, count, start);
        writeHdf5Data(maxDset, maxVals.data(), basicBufferDims, count, start);
        writeHdf5Data(sumDset, sums.data(), basicBufferDims, count, start);
        writeHdf5Data(ssqDset, sumsSq.data(), basicBufferDims, count, start);
        writeHdf5Data(nanDset, nanCounts.data(), basicBufferDims, count, start);
    }
    
    void writeHistogram(const std::vector<hsize_t>& basicBufferDims = EMPTY_DIMS, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS) {
        writeHdf5Data(histDset, histograms.data(), extend(basicBufferDims, {numBins}), count, start);
    }
    
    static hsize_t size(std::vector<hsize_t> dims, hsize_t numBins = 0, hsize_t partialHistMultiplier = 0) {
        auto statsSize = product(dims);
        return (4 * sizeof(double) + sizeof(int64_t)) * statsSize + sizeof(int64_t) * (statsSize * numBins + statsSize * numBins * partialHistMultiplier);
    }
    
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

    // Buffers -- TODO replace with arrays
    std::vector<double> minVals;
    std::vector<double> maxVals;
    std::vector<double> sums;
    std::vector<double> sumsSq;
    std::vector<int64_t> nanCounts;
    
    std::vector<int64_t> histograms;
    std::vector<int64_t> partialHistograms;
};

#endif
