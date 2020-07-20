#ifndef __COMMON_H
#define __COMMON_H

#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <sstream>
#include <limits>
#include <memory>
#include <numeric>
#include <iterator>
#include <cstring>

#include <H5Cpp.h>
#include <fitsio.h>

#define SCHEMA_VERSION "0.3"
#define HDF5_CONVERTER "hdf_convert"
#define HDF5_CONVERTER_VERSION "0.1.13"

#define TILE_SIZE (hsize_t)512
#define MIN_MIPMAP_SIZE (hsize_t)128

#ifdef _VERBOSE_
    #define DEBUG(x) do {x} while (0)
#else
    #define DEBUG(x) do {} while (0)
#endif

#ifdef _TIMER_
    #define TIMER(x) do {x} while (0)
#else
    #define TIMER(x) do {} while (0)
#endif

#define PROGRESS(msg) if (progress) std::cout << msg
#define PROGRESS_DECIMATED(index, stride, msg) if (progress && !(index % stride)) std::cout << msg << std::flush

#define EMPTY_DIMS std::vector<hsize_t>()

#define UNUSED(x) (void)(x)

#endif
