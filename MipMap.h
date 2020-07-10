#ifndef __MIPMAP_H
#define __MIPMAP_H

#include "common.h"
#include "Util.h"

// TODO refactor most of this into a MipMaps parent object

struct MipMap {
    MipMap() {}
    
    MipMap(std::vector<hsize_t> datasetDims, hsize_t width, hsize_t height, hsize_t depth, int divisor) :
        datasetDims(datasetDims),
        divisor(divisor),
        channelSize(width * height),
        width(width),
        height(height),
        depth(depth)
    {}
    
    // TODO pass the sizes in
    void createBuffers() {
        vals.resize(depth * channelSize);
        count.resize(depth * channelSize);
    }
    
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
    
    void resetBuffers() {
        std::fill(vals.begin(), vals.end(), 0);
        std::fill(count.begin(), count.end(), 0);
    }
    
    void createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims) {
        H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
        floatType.setOrder(H5T_ORDER_LE);
        
        std::ostringstream mipMapName;
        mipMapName << "MipMaps/DATA/DATA_XY_" << divisor;
        
        if (useChunks(width, height)) {
            createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims, chunkDims);
        } else {
            createHdf5Dataset(dataset, group, mipMapName.str(), floatType, datasetDims);
        }
    }
    
    void write(hsize_t stokesOffset, hsize_t channelOffset) {
        std::vector<hsize_t> dims = {depth, height, width};
        int N = datasetDims.size();
        std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
        std::vector<hsize_t> start = trimAxes({stokesOffset, channelOffset, 0, 0}, N);
        writeHdf5Data(dataset, vals, dims, count, start);
    }
    
    static void initialise(std::vector<MipMap>& mipMaps, std::vector<hsize_t> dims, hsize_t width, hsize_t height, hsize_t depth) {
        int divisor = 1;
        int N = dims.size();
        while (width > MIN_MIPMAP_SIZE || height > MIN_MIPMAP_SIZE) {
            divisor *= 2;
            width = (width + 1) / 2;
            height = (height + 1) / 2;
            
            dims[N - 1] = width;
            dims[N - 2] = height;
            
            mipMaps.push_back(MipMap(dims, width, height, depth, divisor));
        }
    }
    
    static hsize_t size(hsize_t width, hsize_t height, hsize_t depth) {
        hsize_t size = 0;
        int divisor = 1;
        while (width > MIN_MIPMAP_SIZE || height > MIN_MIPMAP_SIZE) {
            divisor *= 2;
            width = (width + 1) / 2;
            height = (height + 1) / 2;
            size += (sizeof(double) + sizeof(int)) * width * height * depth;
        }
        return size;
    }
    
    std::vector<hsize_t> datasetDims;
    
    H5::DataSet dataset;
    
    int divisor;
    hsize_t channelSize;
    hsize_t width;
    hsize_t height;
    hsize_t depth;
    
    std::vector<double> vals;
    std::vector<int> count;
};

#endif
