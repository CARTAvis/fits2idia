#ifndef __MIPMAP_H
#define __MIPMAP_H

#include "common.h"

struct MipMap {
    MipMap() {}
    
    MipMap(int N, hsize_t width, hsize_t height, hsize_t depth, int divisor) :
        N(N),
        divisor(divisor),
        channelSize(width * height),
        width(width),
        height(height),
        depth(depth),
        vals(depth * channelSize),
        count(depth * channelSize)
    {}
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t totalChannelOffset) {
        hsize_t mipIndex = totalChannelOffset * channelSize + (y / divisor) * width + (x / divisor);
        vals[mipIndex] += val;
        count[mipIndex]++;
    }
    
    void calculate() {
        for (int mipIndex = 0; mipIndex < vals.size(); mipIndex++) {
            if (count[mipIndex]) {
                vals[mipIndex] /= count[mipIndex];
            } else {
                vals[mipIndex] = NAN;
            }
        }
    }
    
    void reset() {
        std::fill(vals.begin(), vals.end(), 0);
        std::fill(count.begin(), count.end(), 0);
    }
    
    bool useChunks() {
        return TILE_SIZE <= width && TILE_SIZE <= height;
    }
    
    void createDataset(H5::Group mipMapGroup, H5::FloatType floatType, Dims dims) {
        std::vector<hsize_t> mipMapDims = dims.mipMapExtra;
        mipMapDims.push_back(height);
        mipMapDims.push_back(width);
        
        H5::DSetCreatPropList createPlist;
        if (useChunks()) {
            createPlist.setChunk(dims.N, dims.tileDims.data());
        }
        
        auto dataSpace = H5::DataSpace(N, mipMapDims.data());
        
        std::ostringstream mipMapName;
        mipMapName << "DATA_XY_" << divisor;
        
        dataSet = mipMapGroup.createDataSet(mipMapName.str().c_str(), floatType, dataSpace, createPlist);
    }
    
    void write(hsize_t stokesOffset, hsize_t channelOffset) {
        std::vector<hsize_t> count;
        std::vector<hsize_t> start;
        
        if (N == 2) {
            count = {height, width};
            start = {0, 0};
        } else if (N == 3) {
            count = {depth, height, width};
            start = {channelOffset, 0, 0};
        } else if (N == 4) {
            count = {1, depth, height, width};
            start = {stokesOffset, channelOffset, 0, 0};
        }
        
        hsize_t memDims[] = {depth, height, width};
        H5::DataSpace memspace(3, memDims);

        auto sliceDataSpace = dataSet.getSpace();
        sliceDataSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

        dataSet.write(vals.data(), H5::PredType::NATIVE_DOUBLE, memspace, sliceDataSpace);
    }
    
    static void initialise(std::vector<MipMap>& mipMaps, int N, hsize_t width, hsize_t height, hsize_t depth) {
        int divisor = 1;
        while (width > MIN_MIPMAP_SIZE || height > MIN_MIPMAP_SIZE) {
            divisor *= 2;
            width = (width + 1) / 2;
            height = (height + 1) / 2;
            mipMaps.push_back(MipMap(N, width, height, depth, divisor));
        }
    }
    
    int N;
    int divisor;
    hsize_t channelSize;
    hsize_t width;
    hsize_t height;
    hsize_t depth;
    
    std::vector<double> vals;
    std::vector<int> count;
    
    H5::DataSet dataSet;
};

#endif
