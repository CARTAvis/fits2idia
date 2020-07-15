#include "Stats.h"

StatsCounter::StatsCounter() : minVal(std::numeric_limits<float>::max()), maxVal(-std::numeric_limits<float>::max()), sum(0), sumSq(0), nanCount(0) {
}

void StatsCounter::accumulateFinite(float val) {
    minVal = fmin(minVal, val);
    maxVal = fmax(maxVal, val);
    sum += val;
    sumSq += val * val;
}

void StatsCounter::accumulateFiniteLazy(float val) {
    if (val < minVal) {
        minVal = val;
    } else if (val > maxVal) {
        maxVal = val;
    }
    sum += val;
    sumSq += val * val;
}

void StatsCounter::accumulateFiniteLazyFirst(float val) {
    minVal = val;
    maxVal = val;
    sum += val;
    sumSq += val * val;
}

void StatsCounter::accumulateNonFinite() {
    nanCount++;
}

void StatsCounter::accumulateStats(Stats stats, hsize_t index) {
    if (std::isfinite(stats.maxVals[index])) {
        sum += stats.sums[index];
        sumSq += stats.sumsSq[index];
        minVal = fmin(minVal, stats.minVals[index]);
        maxVal = fmax(maxVal, stats.maxVals[index]);
    }
    nanCount += stats.nanCounts[index];
}

Stats::Stats(const std::vector<hsize_t>& basicDatasetDims, hsize_t numBins) : basicDatasetDims(basicDatasetDims), numBins(numBins) {}

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
    
    minVals.resize(statsSize, std::numeric_limits<float>::max());
    maxVals.resize(statsSize, -std::numeric_limits<float>::max());
    sums.resize(statsSize);
    sumsSq.resize(statsSize);
    nanCounts.resize(statsSize);
    
    if (numBins) {
        histograms.resize(statsSize * numBins);
        partialHistograms.resize(statsSize * numBins * partialHistMultiplier);
    }
}

void Stats::copyStatsFromCounter(hsize_t index, hsize_t totalVals, const StatsCounter& counter) {
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

void Stats::accumulateHistogram(float val, double min, double range, hsize_t offset) {
    int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
    histograms[offset * numBins + binIndex]++;
}

void Stats::accumulatePartialHistogram(float val, double min, double range, hsize_t offset) {
    int binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - min) / range));
    partialHistograms[offset * numBins + binIndex]++;
}

void Stats::consolidatePartialHistogram() {
    auto multiplier = partialHistograms.size() / histograms.size();
    for (hsize_t offset = 0; offset < multiplier; offset++) {
        for (hsize_t binIndex = 0; binIndex < numBins; binIndex++) {
            histograms[binIndex] += partialHistograms[offset * numBins + binIndex];
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
    writeHdf5Data(minDset, minVals.data(), basicBufferDims, count, start);
    writeHdf5Data(maxDset, maxVals.data(), basicBufferDims, count, start);
    writeHdf5Data(sumDset, sums.data(), basicBufferDims, count, start);
    writeHdf5Data(ssqDset, sumsSq.data(), basicBufferDims, count, start);
    writeHdf5Data(nanDset, nanCounts.data(), basicBufferDims, count, start);
}

void Stats::writeHistogram(const std::vector<hsize_t>& basicBufferDims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    writeHdf5Data(histDset, histograms.data(), extend(basicBufferDims, {numBins}), count, start);
}

void Stats::resetBuffers() {
    std::fill(minVals.begin(), minVals.end(), std::numeric_limits<float>::max());
    std::fill(maxVals.begin(), maxVals.end(), -std::numeric_limits<float>::max());
    std::fill(sums.begin(), sums.end(), 0);
    std::fill(sumsSq.begin(), sumsSq.end(), 0);
    std::fill(nanCounts.begin(), nanCounts.end(), 0);
    
    if (numBins) {
        std::fill(histograms.begin(), histograms.end(), 0);
        std::fill(partialHistograms.begin(), partialHistograms.end(), 0);
    }
}
