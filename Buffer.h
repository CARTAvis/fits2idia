#ifndef __BUFFER_H
#define __BUFFER_H

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
    
    hsize_t size;
    T* data;
};



#endif
