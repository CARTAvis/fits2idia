/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Stats.h"

Stats::Stats() : basicDatasetDims({}), numBins(0), partialHistMultiplier(0), buffersAllocated(0), histogramBuffersAllocated(0) {}

Stats::Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins) : basicDatasetDims(basicDatasetDims), numBins(numBins), partialHistMultiplier(0), buffersAllocated(0), histogramBuffersAllocated(0) {}

Stats::~Stats() {
    if (buffersAllocated) {
        delete[] minVals;
        delete[] maxVals;
        delete[] sums;
        delete[] sumsSq;
        delete[] nanCounts;
        if (histogramBuffersAllocated) {
            delete[] histograms;
            delete[] partialHistograms;
        }
    }
}

hsize_t Stats::size(std::vector<hsize_t> dims, hsize_t numBins, hsize_t partialHistMultiplier) {
    auto statsSize = product(dims);
    DEBUG(std::cout << "DEBUG: Stats::size statsSize = " << statsSize << " , numBins = " << numBins << " , partialHistMultiplier = " << partialHistMultiplier << std::endl;);
    return (2 * sizeof(float) + 2 * sizeof(double) + sizeof(int64_t)) * statsSize + sizeof(int64_t) * (statsSize * numBins + statsSize * numBins * partialHistMultiplier);
}

void Stats::createDatasets(H5::Group group, std::string name, const std::vector<hsize_t>& chunkDims /* = {}*/) {
    H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
    floatType.setOrder(H5T_ORDER_LE);
    
    H5::IntType intType(H5::PredType::NATIVE_INT64);
    intType.setOrder(H5T_ORDER_LE);
    
    createHdf5Dataset(minDset, group, "Statistics/" + name + "/MIN", floatType, basicDatasetDims, chunkDims);
    createHdf5Dataset(maxDset, group, "Statistics/" + name + "/MAX", floatType, basicDatasetDims, chunkDims);
    createHdf5Dataset(sumDset, group, "Statistics/" + name + "/SUM", floatType, basicDatasetDims, chunkDims);
    createHdf5Dataset(ssqDset, group, "Statistics/" + name + "/SUM_SQ", floatType, basicDatasetDims, chunkDims);
    createHdf5Dataset(nanDset, group, "Statistics/" + name + "/NAN_COUNT", intType, basicDatasetDims, chunkDims);
    
    if (numBins) {
        createHdf5Dataset(histDset, group, "Statistics/" + name + "/HISTOGRAM", intType, extend(basicDatasetDims, {numBins}), chunkDims.empty() ? chunkDims : extend(chunkDims, {numBins}));
    }
}

void Stats::createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier) {
    fullBasicBufferDims = dims;
    auto statsSize = product(dims);
        
    long int total_bytes = 2*statsSize*sizeof(float) + 2*statsSize*sizeof(double) + statsSize*sizeof(int64_t);

    minVals = new float[statsSize];
    maxVals = new float[statsSize];
    sums = new double[statsSize];
    sumsSq = new double[statsSize];
    nanCounts = new int64_t[statsSize];
    buffersAllocated = true;

    std::cout << "MEMORY (Stats::createBuffers): starting with total_bytes = " << total_bytes << " bytes." << std::endl;
    
    if (numBins) {
        histograms = new int64_t[statsSize * numBins];
        partialHistograms = new int64_t[statsSize * numBins * partialHistMultiplier];
        this->partialHistMultiplier = partialHistMultiplier;
        histogramBuffersAllocated = true;
        size_t histogram_bytes = (statsSize * numBins * partialHistMultiplier + statsSize*numBins)*sizeof(int64_t);
        total_bytes += histogram_bytes;
        std::cout << "MEMORY (Stats::createBuffers): adding " << histogram_bytes << " bytes for histograms" << std::endl;
    }    
    
    std::cout << "DEBUG : Stats::createBuffers , partialHistMultiplier = " << partialHistMultiplier << " statsSize = " << statsSize << " numBins = " << numBins << std::endl << std::flush;
    std::cout << "MEMORY (Stats::createBuffers): allocating " << double(total_bytes)/1e9 << " GB " << std::endl << std::flush;

}

void Stats::copyHistogramBuffers(const Stats& right) {
   if (histogramBuffersAllocated) {
        auto statsSize = product(fullBasicBufferDims);
        memcpy(histograms, right.histograms, sizeof(int64_t) * statsSize * numBins);
        memcpy(partialHistograms, right.partialHistograms, sizeof(int64_t) * statsSize * numBins * partialHistMultiplier);
    }
}

void Stats::clearHistogramBuffers() {
    if (histogramBuffersAllocated) {
        auto statsSize = product(fullBasicBufferDims);
        memset(histograms, 0, sizeof(int64_t) * statsSize * numBins);
        memset(partialHistograms, 0, sizeof(int64_t) * statsSize * numBins * partialHistMultiplier);
    }
}

void Stats::write() {
    writeBasic(fullBasicBufferDims);
    
    if (numBins) {
        writeHistogram(fullBasicBufferDims);
    }
}

void Stats::write(const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    write(fullBasicBufferDims, count, start);
}

void Stats::write(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    auto basicN = basicDatasetDims.size();
    writeBasic(basicBufferDims, trimAxes(count, basicN), trimAxes(start, basicN));
    
    if (numBins) {
        auto histN = basicN + 1;
        writeHistogram(basicBufferDims, trimAxes(extend(count, {numBins}), histN), trimAxes(extend(start, {0}), histN));
    }
}
    
void Stats::writeBasic(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    writeHdf5Data(minDset, minVals, basicBufferDims, count, start);
    writeHdf5Data(maxDset, maxVals, basicBufferDims, count, start);
    writeHdf5Data(sumDset, sums, basicBufferDims, count, start);
    writeHdf5Data(ssqDset, sumsSq, basicBufferDims, count, start);
    writeHdf5Data(nanDset, nanCounts, basicBufferDims, count, start);
}

void Stats::writeHistogram(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    writeHdf5Data(histDset, histograms, extend(basicBufferDims, {numBins}), count, start);
}
