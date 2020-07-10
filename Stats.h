#ifndef __STATS_H
#define __STATS_H

#include "common.h"
#include "Util.h"

struct Stats {
    Stats() {}
    
    Stats(const std::vector<hsize_t>& statsDims, const std::vector<hsize_t>& histDims = EMPTY_DIMS) : statsDims(statsDims), histDims(histDims) {}
    
    void createDatasets(H5::Group group, std::string name) {
        H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
        floatType.setOrder(H5T_ORDER_LE);
        
        H5::IntType intType(H5::PredType::NATIVE_INT64);
        intType.setOrder(H5T_ORDER_LE);
        
        createHdf5Dataset(minDset, group, "Statistics/" + name + "/MIN", floatType, statsDims);
        createHdf5Dataset(maxDset, group, "Statistics/" + name + "/MAX", floatType, statsDims);
        createHdf5Dataset(sumDset, group, "Statistics/" + name + "/SUM", floatType, statsDims);
        createHdf5Dataset(ssqDset, group, "Statistics/" + name + "/SUM_SQ", floatType, statsDims);
        createHdf5Dataset(nanDset, group, "Statistics/" + name + "/NAN_COUNT", intType, statsDims);
        
        if (!histDims.empty()) {
            createHdf5Dataset(histDset, group, "Statistics/" + name + "/HISTOGRAM", intType, histDims);
        }
    }
    
    void createBuffers(hsize_t statsSize, int numBins = 0, hsize_t partialHistMultiplier = 0) {
        minVals.resize(statsSize, std::numeric_limits<double>::max());
        maxVals.resize(statsSize, -std::numeric_limits<double>::max());
        sums.resize(statsSize);
        sumsSq.resize(statsSize);
        nanCounts.resize(statsSize);
        
        histograms.resize(statsSize * numBins);
        partialHistograms.resize(statsSize * numBins * partialHistMultiplier);
    }
    
    void resetBuffers() {
        std::fill(minVals.begin(), minVals.end(), std::numeric_limits<double>::max());
        std::fill(maxVals.begin(), maxVals.end(), -std::numeric_limits<double>::max());
        std::fill(sums.begin(), sums.end(), 0);
        std::fill(sumsSq.begin(), sumsSq.end(), 0);
        std::fill(nanCounts.begin(), nanCounts.end(), 0);
        
        std::fill(histograms.begin(), histograms.end(), 0);
        std::fill(partialHistograms.begin(), partialHistograms.end(), 0);
    }
    
    // TODO eventually add helper write functions for translating tile / swizzled slice offsets to count and start
    
    void writeBasic(const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS) {
        writeHdf5Data(minDset, minVals.data(), dims, count, start);
        writeHdf5Data(maxDset, maxVals.data(), dims, count, start);
        writeHdf5Data(sumDset, sums.data(), dims, count, start);
        writeHdf5Data(ssqDset, sumsSq.data(), dims, count, start);
        writeHdf5Data(nanDset, nanCounts.data(), dims, count, start);
    }
    
    void writeHistogram(const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS) {
        writeHdf5Data(histDset, histograms.data(), dims, count, start);
    }
    
    static hsize_t size(hsize_t statsSize, int numBins = 0, hsize_t partialHistMultiplier = 0) {
        return (4 * sizeof(double) + sizeof(int64_t)) * statsSize + sizeof(int64_t) * (statsSize * numBins + statsSize * numBins * partialHistMultiplier);
    }
    
    // Dataset dimensions
    std::vector<hsize_t> statsDims;
    std::vector<hsize_t> histDims;
    
    // Datasets
    H5::DataSet minDset;
    H5::DataSet maxDset;
    H5::DataSet sumDset;
    H5::DataSet ssqDset;
    H5::DataSet nanDset;
    
    H5::DataSet histDset;

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
