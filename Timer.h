#ifndef __TIMER_H
#define __TIMER_H

#include "common.h"
#include <unordered_map>

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
    Timer() : activeCounter() {}
    
    std::unordered_map<std::string, TimerCounter> counters;
    TimerCounter* activeCounter;
    
    void start(std::string label) {
        if (activeCounter) {
            activeCounter->stop();
        }
        activeCounter = &counters[label];
        activeCounter->start();
    }
    
    void print(hsize_t imageSize) {
        if (activeCounter) {
            activeCounter->stop();
        }
        std::cout << std::endl;
        TimerCounter total;
        for (auto& c : counters) {
            auto& label = c.first;
            auto& counter = c.second;
            total = total + counter;
            std::cout << label << ": " << counter.seconds() << " seconds (" << counter.speed(imageSize) << " MB/s)" << std::endl;
        }
        std::cout << "TOTAL: " << total.seconds() << " seconds (" << total.speed(imageSize) << " MB/s)" << std::endl;
    }
};

#endif
