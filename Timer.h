#ifndef __TIMER_H
#define __TIMER_H

#include "common.h"

struct TimerCounter {
    TimerCounter() : value(0) {}
    
    TimerCounter(unsigned int value) : value(value) {}
    
    TimerCounter operator+(const TimerCounter& other) {
        return TimerCounter(this->value + other.value);
    }
    
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        auto stopTime = std::chrono::high_resolution_clock::now();
        value += std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count();
    }
    
    double seconds() {
        return value * 1e-3;
    }
    
    double speed(hsize_t imageSize) {
        return (imageSize * 4) * 1.0e-6 / seconds();
    }
    
    unsigned int value;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

struct Timer {
    Timer() {}
    Timer(hsize_t imageSize, bool slow) : 
        size(imageSize),
        loopLabel1(slow ? "XY, XYZ & Z stats" : "Swizzling, XY & XYZ stats"),
        loopLabel2(slow ? "Histograms" : "Z stats"),
        loopLabel3(slow ? "Swizzling" : "Histograms"),
        loopLabel4(slow ? "????" : "MipMaps")
    {}
    
    hsize_t size;
    
    TimerCounter alloc;
    TimerCounter read;
    TimerCounter process1;
    TimerCounter process2;
    TimerCounter process3;
    TimerCounter process4;
    TimerCounter write;
    
    std::string loopLabel1;
    std::string loopLabel2;
    std::string loopLabel3;
    std::string loopLabel4;
    
    void print() {
        TimerCounter process = process1 + process2 + process3 + process4;
        TimerCounter total = alloc + read + process + write;
        
        std::cout << std::endl;
        std::cout << "Allocated in " << alloc.seconds() << " seconds (" << alloc.speed(size) << " MB/s)" << std::endl;
        std::cout << "Read in " << read.seconds() << " seconds (" << read.speed(size) << " MB/s)" << std::endl;
        std::cout << "Processed in " << process.seconds() << " seconds (" << process.speed(size) << " MB/s)" << std::endl;
        std::cout << "\t" << loopLabel1 << ": " << process1.seconds() << std::endl;
        std::cout << "\t" << loopLabel2 << ": " << process2.seconds() << std::endl;
        std::cout << "\t" << loopLabel3 << ": " << process3.seconds() << std::endl;
        std::cout << "\t" << loopLabel4 << ": " << process4.seconds() << std::endl;
        std::cout << "Written in " << write.seconds() << " seconds (" << write.speed(size) << " MB/s)" << std::endl;
        std::cout << "TOTAL: " << total.seconds() << " seconds (" << total.speed(size) << " MB/s)" << std::endl;
    }
};

#endif
