#ifndef __MIPMAP_H
#define __MIPMAP_H

#include "common.h"
#include "Util.h"

// A single mipmap
struct MipMap {
    MipMap() {};
    MipMap(const std::vector<hsize_t>& datasetDims, int mip);
    
    void createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims);
    void createBuffers(std::vector<hsize_t>& bufferDims);
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t totalChannelOffset = 0);
    void calculate();
    void write(hsize_t stokesOffset, hsize_t channelOffset);
    void resetBuffers();
    
    std::vector<hsize_t> datasetDims;
    int mip;
    
    H5::DataSet dataset;
    
    std::vector<hsize_t> bufferDims;
    hsize_t width;
    hsize_t height;
    hsize_t depth;
    hsize_t stokes;
    
    std::vector<double> vals;
    std::vector<int> count;
};

// A set of mipmaps
struct MipMaps {
    MipMaps() {};
    MipMaps(std::vector<hsize_t> standardDims, const std::vector<hsize_t>& chunkDims);
    
    // We need the dataset dimensions to work out how many mipmaps we have
    static hsize_t size(const std::vector<hsize_t>& standardDims, const std::vector<hsize_t>& standardBufferDims);
    
    void createDatasets(H5::Group group);
    void createBuffers(const std::vector<hsize_t>& standardBufferDims);
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t totalChannelOffset = 0);
    void calculate();
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
