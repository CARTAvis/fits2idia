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

    // WARNING: copy constructor is created only to copy MipMap structure by firstprivate(thread_MipMaps) 
    //          and create all the buffers of the same size and length as originally. Hence, vals and count arrays are not memcpy-ied 
// WARNING / ERROR : copy constructor crashes push_back(MipMaps(...)) - this should really be fixed at some point !!!
/*    MipMap(const MipMap& right): mipXY(right.mipXY), mipZ(right.mipZ), bufferSize(right.bufferSize), width(right.width), height(right.height), 
       depth(right.depth), stokes(right.stokes) {
       datasetDims = right.datasetDims;
       bufferDims  = right.bufferDims;
       // dataset is ignored here, we do not want to deal with copying HDF5 file structures, groups etc
       
       std::cout << "DEBUG : copy constructor : before createBuffers( bufferDims )" << std::endl;
       createBuffers( bufferDims );

       std::cout << "DEBUG : copy constructor of MipMap called" << std::endl;
    }*/
    ~MipMap();
    
    void createDataset(H5::Group group, const std::vector<hsize_t>& chunkDims);
    void createBuffers(std::vector<hsize_t>& bufferDims);
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t z) {
        hsize_t mipIndex = (z / mipZ) * width * height + (y / mipXY) * width + (x / mipXY);
        vals[mipIndex] += val;
        count[mipIndex]++;
    }
    
    void accumulateFromMipMap(const MipMap& mipmap) {
        if (bufferSize != mipmap.bufferSize ) {
           std::cerr << "ERROR in accumulateFromMipMap different buffer sizes " << bufferSize << " != " << mipmap.bufferSize << std::endl;
        }        
    
        for(hsize_t mipIndex=0;mipIndex<bufferSize;mipIndex++){
           vals[mipIndex] += mipmap.vals[mipIndex];
           count[mipIndex] += mipmap.count[mipIndex];
        }
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
    
    double* vals; // size : bufferSize
    int* count;   // size : bufferSize 
};

// A set of mipmaps
struct MipMaps {
    MipMaps() {};
    MipMaps(std::vector<hsize_t> standardDims, const std::vector<hsize_t>& chunkDims, bool zMips);
    
    // We need the dataset dimensions to work out how many mipmaps we have
    static hsize_t size(const std::vector<hsize_t>& standardDims, const std::vector<hsize_t>& standardBufferDims, bool zMips);
    
    
    void createDatasets(H5::Group group);
    void createBuffers(const std::vector<hsize_t>& standardBufferDims);
    
    void accumulate(double val, hsize_t x, hsize_t y, hsize_t z) {
        for (auto& mipMap : mipMaps) {
            mipMap.accumulate(val, x, y, z);
        }
    }

    void accumulateFromMipMaps(const MipMaps& mipmaps) {
//        for (auto& mipMap : mipMaps) {
//            mipMap.accumulate(val, x, y, z);
//        }
        
        if (mipMaps.size() != mipmaps.mipMaps.size() ) {
           std::cerr << "ERROR in accumulateFromMipMaps : different number of MipMaps " << mipMaps.size() << " != " << mipmaps.mipMaps.size() << std::endl;
        }
        
        for (int m=0;m<mipMaps.size();m++){
           MipMap& mipMap = mipMaps[m];
           const MipMap& mipMap2 = mipmaps.mipMaps[m];
           
           // TODO : check if dimensions agree !!!
           mipMap.accumulateFromMipMap(mipMap2);           
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
