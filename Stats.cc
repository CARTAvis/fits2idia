#include "Stats.h"

Stats::Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins) : basicDatasetDims(basicDatasetDims), numBins(numBins), partialHistMultiplier(0) {}

Stats::~Stats() {
    if (!fullBasicBufferDims.empty()) {
        delete[] minVals;
        delete[] maxVals;
        delete[] sums;
        delete[] sumsSq;
        delete[] nanCounts;
        if (numBins) {
            delete[] histograms;
            delete[] partialHistograms;
        }
    }
}

hsize_t Stats::size(std::vector<hsize_t> dims, hsize_t numBins, hsize_t partialHistMultiplier) {
    auto statsSize = product(dims);
    return (2 * sizeof(float) + 2 * sizeof(double) + sizeof(int64_t)) * statsSize + sizeof(int64_t) * (statsSize * numBins + statsSize * numBins * partialHistMultiplier);
}

void Stats::createDatasets(H5::Group group, std::string name) {
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

void Stats::createBuffers(std::vector<hsize_t> dims, hsize_t partialHistMultiplier) {
    fullBasicBufferDims = dims;
    auto statsSize = product(dims);
    
    minVals = new float[statsSize];
    maxVals = new float[statsSize];
    sums = new double[statsSize];
    sumsSq = new double[statsSize];
    nanCounts = new int64_t[statsSize];
    
    if (numBins) {
        histograms = new int64_t[statsSize * numBins];
        partialHistograms = new int64_t[statsSize * numBins * partialHistMultiplier];
        this->partialHistMultiplier = partialHistMultiplier;
    }
}

void Stats::clearHistogramBuffers() {
    auto statsSize = product(fullBasicBufferDims);
    for (hsize_t i = 0; i < statsSize * numBins; i++) {
        histograms[i] = 0;
        for (hsize_t j = 0; j < partialHistMultiplier; j++) {
            partialHistograms[partialHistMultiplier * i + j] = 0;
        }
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
