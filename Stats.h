#ifndef __STATS_H
#define __STATS_H

#include "common.h"

struct Stats {
    Stats() {}
    
    Stats(StatsDims dims) : 
        dims(dims),
        minVals(dims.statsSize, std::numeric_limits<double>::max()), 
        maxVals(dims.statsSize, -std::numeric_limits<double>::max()),
        sums(dims.statsSize),
        sumsSq(dims.statsSize),
        nanCounts(dims.statsSize),
        histograms(dims.histSize),
        partialHistograms(dims.partialHistSize)
    {}
        
    void writeDset(H5::Group& group, std::string name, std::vector<double>& vals, H5::FloatType file_dtype, H5::DataSpace dspace) {
        auto dset = group.createDataSet(name, file_dtype, dspace);
        dset.write(vals.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void writeDset(H5::Group& group, std::string name, std::vector<int64_t>& vals, H5::IntType file_dtype, H5::DataSpace dspace) {
        auto dset = group.createDataSet(name, file_dtype, dspace);
        dset.write(vals.data(), H5::PredType::NATIVE_INT64);
    }

    void write(H5::Group& group, H5::FloatType floatType, H5::IntType intType) {
        auto statsSpace = H5::DataSpace(dims.statsDims.size(), dims.statsDims.data());
        writeDset(group, "MIN", minVals, floatType, statsSpace);
        writeDset(group, "MAX", maxVals, floatType, statsSpace);
        writeDset(group, "SUM", sums, floatType, statsSpace);
        writeDset(group, "SUM_SQ", sumsSq, floatType, statsSpace);
        writeDset(group, "NAN_COUNT", nanCounts, intType, statsSpace);
        
        if (dims.histSize) {
            auto histSpace = H5::DataSpace(dims.histDims.size(), dims.histDims.data());
            writeDset(group, "HISTOGRAM", histograms, intType, histSpace);
        }
    }
    
    static hsize_t size(StatsDims dims) {
        return (4 * sizeof(double) + sizeof(int64_t)) * dims.statsSize + sizeof(int64_t) * (dims.histSize + dims.partialHistSize);
    }
    
    StatsDims dims;

    std::vector<double> minVals;
    std::vector<double> maxVals;
    std::vector<double> sums;
    std::vector<double> sumsSq;
    std::vector<int64_t> nanCounts;
    std::vector<int64_t> histograms;
    std::vector<int64_t> partialHistograms;
};

#endif
