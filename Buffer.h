#ifndef __DIMS_H
#define __DIMS_H

#include "common.h"

template <typename T>
struct Buffer {
    Buffer() {}
    
    Buffer(hsize_t size) : size(size) {
        data = new T[size];
    }
    
    Buffer(hsize_t size, T defaultValue) : Buffer(size) {
        reset(defaultValue);
    }
    
    ~Buffer() {
        if (data) {
            delete data;
        }
    }
    
    T& operator[](int index) {
        return data[index];
    }
    
    void reset(T value) {
        for (hsize_t i = 0; i < size; i++) {
            data[i] = defaultValue;
        }
    }
    
    void write(H5::DataSet destination, hsize_t offset) {
        // TODO how to determine types?
    }
    
    hsize_t size;
    T* data;
};



#endif
