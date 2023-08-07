/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "MipMap.h"

// MipMap

MipMap::MipMap(const std::vector<hsize_t>& datasetDims, int mipXY, int mipZ) : datasetDims(datasetDims), mipXY(mipXY), mipZ(mipZ) {}

MipMap::~MipMap() {
    if (!bufferDims.empty()) {
        delete[] vals;
        delete[] count;
    }
}

void MipMap::createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims) {
    H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
    floatType.setOrder(H5T_ORDER_LE);
    
    std::ostringstream mipMapName;
    
    mipMapName << "MipMaps/DATA/DATA_";
    
    if (mipXY == mipZ)
        mipMapName << "XYZ_" << mipXY;
    else if (mipXY < 2)
        mipMapName << "Z_" << mipZ;
    else if (mipZ < 2)
        mipMapName << "XY_" << mipXY;
    else
        mipMapName << "XYZ_" << mipXY << "_" << mipXY << "_" << mipZ;
    
    if (useChunks(datasetDims)) {
        createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims, chunkDims);
    } else {
        createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims);
    }
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

void MipMap::write(hsize_t stokesOffset, hsize_t channelOffset) {
    int N = datasetDims.size();
    std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
    std::vector<hsize_t> start = trimAxes({stokesOffset, channelOffset, 0, 0}, N);
    
    writeHdf5Data(dataset, vals, bufferDims, count, start);
}

void MipMap::resetBuffers() {
    memset(vals, 0, sizeof(double) * bufferSize);
    memset(count, 0, sizeof(int) * bufferSize);
}

// MipMaps

MipMaps::MipMaps(std::vector<hsize_t> standardDims, const std::vector<hsize_t>& chunkDims, bool zMips) : standardDims(standardDims), chunkDims(chunkDims) {
    auto dims = standardDims;
    int N = dims.size();
    int mipXY = 1;
    int mipZ = 1;
    
    // We keep going until we have a mipmap which fits entirely within the minimum size
    do {
        if (N > 2 && zMips) {
            do {
                if (mipXY > 1 || mipZ > 1)
                    mipMaps.push_back(MipMap(dims, mipXY, mipZ));
                mipZ *= 2;
                dims = mipDims(dims, 1, 2);
            } while (2 * dims[N - 3] > MIN_MIPMAP_SIZE);
            mipZ = 1;
            dims[N - 3] = standardDims[N - 3];
        } else if (mipXY > 1)
            mipMaps.push_back(MipMap(dims, mipXY, mipZ));
        mipXY *= 2;
        dims = mipDims(dims, 2, 1);
    } while (2 * dims[N - 1] > MIN_MIPMAP_SIZE || 2 * dims[N - 2] > MIN_MIPMAP_SIZE);

}

hsize_t MipMaps::size(const std::vector<hsize_t>& standardDims, const std::vector<hsize_t>& standardBufferDims) {
    hsize_t size = 0;
    int mipXY = 1;
    int mipZ = 1;
    auto datasetDims = standardDims;
    auto bufferDims = standardBufferDims;
    int N = standardDims.size();
    
    do {
        if (N > 2) {
            do {
                if (mipXY > 1 || mipZ > 1)
                    size += (sizeof(double) + sizeof(int)) * product(bufferDims);
                mipZ *= 2;
                datasetDims = mipDims(datasetDims, 1, 2);
                bufferDims = mipDims(bufferDims, 1, 2);
            } while (2 * datasetDims[N - 3] > MIN_MIPMAP_SIZE);
            mipZ = 1;
            datasetDims[N - 3] = standardDims[N - 3];
            bufferDims[N - 3] = standardBufferDims[N - 3];
        } else if (mipXY > 1)
            size += (sizeof(double) + sizeof(int)) * product(bufferDims);
        mipXY *= 2;
        datasetDims = mipDims(datasetDims, 2, 1);
        bufferDims = mipDims(bufferDims, 2, 1);
    } while (2 * datasetDims[N - 1] > MIN_MIPMAP_SIZE || 2 * datasetDims[N - 2] > MIN_MIPMAP_SIZE);

    return size;
}

void MipMaps::createDatasets(H5::Group group) {
    for (auto& mipMap : mipMaps) {
        mipMap.createDataset(group, chunkDims);
    }
    
}

void MipMaps::createBuffers(const std::vector<hsize_t>& standardBufferDims) {
    for (auto& mipMap : mipMaps) {
        auto dims = mipDims(standardBufferDims, mipMap.mipXY, mipMap.mipZ);
        mipMap.createBuffers(dims);
    }
}

void MipMaps::write(hsize_t stokesOffset, hsize_t channelOffset) {
    for (auto& mipMap : mipMaps) {
        mipMap.write(stokesOffset, channelOffset);
    }
}

void MipMaps::resetBuffers() {
    for (auto& mipMap : mipMaps) {
        mipMap.resetBuffers();
    }
}
