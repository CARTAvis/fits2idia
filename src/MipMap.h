/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef __MIPMAP_H
#define __MIPMAP_H

#include "common.h"
#include "Util.h"

// A single mipmap
struct MipMap {
    MipMap() {};
    MipMap(const std::vector<hsize_t>& datasetDims, int mipXY, int mipZ);
    ~MipMap();
    
    void createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims);
    void createBuffers(std::vector<hsize_t>& bufferDims);
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t z) {
        hsize_t mipIndex = (z / mipZ) * width * height + (y / mipXY) * width + (x / mipXY);
        vals[mipIndex] += val;
        count[mipIndex]++;
    }
    
    void calculate() {
        for (hsize_t mipIndex = 0; mipIndex < bufferSize; mipIndex++) {
            if (count[mipIndex]) {
                vals[mipIndex] /= count[mipIndex];
            } else {
                vals[mipIndex] = NAN;
            }
        }
    }
    
    void write(hsize_t stokesOffset, hsize_t channelOffset);
    void resetBuffers();
    
    std::vector<hsize_t> datasetDims;
    int mipXY;
    int mipZ;
    
    H5::DataSet dataset;
    
    std::vector<hsize_t> bufferDims;
    hsize_t bufferSize;
    
    hsize_t width;
    hsize_t height;
    hsize_t depth;
    hsize_t stokes;
    
    double* vals;
    int* count;
};

// A set of mipmaps
struct MipMaps {
    MipMaps() {};
    MipMaps(std::vector<hsize_t> standardDims, const std::vector<hsize_t>& chunkDims, bool zMips);
    
    // We need the dataset dimensions to work out how many mipmaps we have
    static hsize_t size(const std::vector<hsize_t>& standardDims, const std::vector<hsize_t>& standardBufferDims);
    
    void createDatasets(H5::Group group);
    void createBuffers(const std::vector<hsize_t>& standardBufferDims);
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t z) {
        for (auto& mipMap : mipMaps) {
            mipMap.accumulate(val, x, y, z);
        }
    }

    void calculate() {
        for (auto& mipMap : mipMaps) {
            mipMap.calculate();
        }
    }
    
    // TODO if we ever want a tiled mipmap calculation
    // we'll need to implement options to pass in custom buffer dims
    // and additional x and y offsets
    void write(hsize_t stokesOffset, hsize_t channelOffset);
    void resetBuffers();
    
    std::vector<hsize_t> standardDims;
    std::vector<hsize_t> chunkDims;
    
    std::vector<MipMap> mipMaps;
};

#endif
