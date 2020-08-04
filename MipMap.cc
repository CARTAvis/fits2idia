#include "MipMap.h"

// MipMap

MipMap::MipMap(const std::vector<hsize_t>& datasetDims, int mip) : datasetDims(datasetDims), mip(mip) {}

MipMap::~MipMap() {
    if (!bufferDims.empty()) {
        delete[] vals;
        delete[] count;
    }
}

// void MipMap::createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims) {
//     H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
//     floatType.setOrder(H5T_ORDER_LE);
//
//     std::ostringstream mipMapName;
//     mipMapName << "MipMaps/DATA/DATA_XY_" << mip;
//
//     if (useChunks(datasetDims)) {
//         createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims, chunkDims);
//     } else {
//         createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims);
//     }
//
// }

void MipMap::createDataset(H5OutputFile &H5outputfile, std::string path, const std::vector<hsize_t>& chunkDims) {

    std::ostringstream mipMapName;
    mipMapName << "/MipMaps/DATA/DATA_XY_" << mip;
    std::vector<hsize_t> chunks = chunkDims;
    datasetfullpath = path+mipMapName.str();
    if (~useChunks(datasetDims)) chunks.clear();
    H5outputfile.create_dataset(datasetfullpath, H5T_NATIVE_FLOAT, datasetDims, chunks);
}

void MipMap::createBuffers(std::vector<hsize_t>& bufferDims) {
    bufferSize = product(bufferDims);

    vals = new double[bufferSize];
    count = new int[bufferSize];

    resetBuffers();

    this->bufferDims = bufferDims;

    auto N = bufferDims.size();

    width = bufferDims[N - 1];
    height = bufferDims[N - 2];
    depth = N > 2 ? bufferDims[N - 3] : 1;
    stokes = N > 3 ? bufferDims[N - 4] : 1;
}

void MipMap::write(H5OutputFile &H5outputfile, hsize_t stokesOffset, hsize_t channelOffset) {
    int N = datasetDims.size();
    std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
    std::vector<hsize_t> start = trimAxes({stokesOffset, channelOffset, 0, 0}, N);
    // writeHdf5Data(dataset, vals, bufferDims, count, start);
    H5outputfile.write_to_dataset_nd(datasetfullpath, bufferDims, vals, count, start);
}

void MipMap::resetBuffers() {
    memset(vals, 0, sizeof(double) * bufferSize);
    memset(count, 0, sizeof(int) * bufferSize);
}

// MipMaps

MipMaps::MipMaps(std::vector<hsize_t> standardDims, const std::vector<hsize_t>& chunkDims) : standardDims(standardDims), chunkDims(chunkDims) {
    auto dims = standardDims;
    int N = dims.size();
    int mip = 1;

    // We keep going until we have a mipmap which fits entirely within the minimum size
    while (dims[N - 1] > MIN_MIPMAP_SIZE || dims[N - 2] > MIN_MIPMAP_SIZE) {
        mip *= 2;
        dims = mipDims(dims, 2);
        mipMaps.push_back(MipMap(dims, mip));
    }
}

hsize_t MipMaps::size(const std::vector<hsize_t>& standardDims, const std::vector<hsize_t>& standardBufferDims) {
    hsize_t size = 0;
    int mip = 1;
    auto datasetDims = standardDims;
    auto bufferDims = standardBufferDims;
    int N = standardDims.size();

    while (datasetDims[N - 1] > MIN_MIPMAP_SIZE || datasetDims[N - 2] > MIN_MIPMAP_SIZE) {
        mip *= 2;
        datasetDims = mipDims(datasetDims, 2);
        bufferDims = mipDims(bufferDims, 2);
        size += (sizeof(double) + sizeof(int)) * product(bufferDims);
    }

    return size;
}

// void MipMaps::createDatasets(H5::Group group) {
//     for (auto& mipMap : mipMaps) {
//         mipMap.createDataset(group, chunkDims);
//     }
//
// }

void MipMaps::createDatasets(H5OutputFile &H5outputfile, std::string path) {
    for (auto& mipMap : mipMaps) {
        mipMap.createDataset(H5outputfile, path, chunkDims);
    }
}

void MipMaps::createBuffers(const std::vector<hsize_t>& standardBufferDims) {
    for (auto& mipMap : mipMaps) {
        auto dims = mipDims(standardBufferDims, mipMap.mip);
        mipMap.createBuffers(dims);
    }
}

void MipMaps::write(H5OutputFile &H5outputfile, hsize_t stokesOffset, hsize_t channelOffset) {
    for (auto& mipMap : mipMaps) {
        mipMap.write(H5outputfile, stokesOffset, channelOffset);
    }
}

void MipMaps::resetBuffers() {
    for (auto& mipMap : mipMaps) {
        mipMap.resetBuffers();
    }
}
