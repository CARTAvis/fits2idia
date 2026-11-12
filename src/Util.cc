/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Util.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/resource.h>

std::vector<std::string> split(const std::string &str, char separator) {
    std::vector<std::string> result;
    std::istringstream stream(str);
    std::string substr;
    while (std::getline(stream, substr, separator)) {
        result.push_back(substr);
    }
    return result;
}

std::vector<hsize_t> trimAxes(const std::vector<hsize_t>& dims, int N) {
    auto trimmed = dims;
    while (trimmed.size() > (unsigned int)N) {
        
        trimmed.erase(trimmed.begin());
    }
    return trimmed;
}

std::vector<hsize_t> extend(const std::vector<hsize_t>& left, const std::vector<hsize_t>& right) {
    std::vector<hsize_t> result;
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    return result;
}

std::vector<hsize_t> mipDims(const std::vector<hsize_t>& dims, int mipXY, int mipZ) {
    int N = dims.size();
    auto mipDims = dims;
    if (mipXY > 1) {
        for (auto i = std::max(0, N - 2); i < N; i++) {
            mipDims[i] = std::ceil((float)mipDims[i] / mipXY);
        }
    }
    if (mipZ > 1 && N > 2)
        mipDims[N - 3] = std::ceil((float)mipDims[N - 3] / mipZ);
    return mipDims;
}

hsize_t product(const std::vector<hsize_t>& dims) {
    return std::accumulate(begin(dims), end(dims), (hsize_t)1, std::multiplies<hsize_t>());
}

bool useChunks(const std::vector<hsize_t>& dims) {    
    int N = dims.size();
    
    for (auto i = std::max(0, N - 2); i < N; i++) {
        if (dims[i] < TILE_SIZE) {
            return false;
        }
    }
    
    return true;
}

void openFitsFile(fitsfile** filePtrPtr, const std::string& fileName) {
    int status(0);
    
    fits_open_file(filePtrPtr, fileName.c_str(), READONLY, &status);
    
    if (status != 0) {
        throw "Could not open FITS file";
    }
    
    int bitpix;
    fits_get_img_type(*filePtrPtr, &bitpix, &status);
    
    if (status != 0) {
        throw "Could not read image type";
    }

    if (bitpix != -32) {
        throw "Currently only supports FP32 files";
    }
}

void closeFitsFile(fitsfile* filePtr) {
    int status(0);
    
    fits_close_file(filePtr, &status);
    
    if (status != 0) {
        throw "Could not close FITS file";
    }
}

void getFitsDims(fitsfile* filePtr, int& N, long* dims) {
    int status(0);
    
    fits_get_img_dim(filePtr, &N, &status);
    
    if (status != 0) {
        throw "Could not read image dimensions";
    }

    if (N < 2 || N > 4) {
        throw "Currently only supports 2D, 3D and 4D cubes";
    }
    
    fits_get_img_size(filePtr, 4, dims, &status);
    
    if (status != 0) {
        throw "Could not read image size";
    }
}

void readFitsHeader(fitsfile* filePtr, int& numAttributes) {
    int status(0);
    
    fits_get_hdrspace(filePtr, &numAttributes, NULL, &status);
    
    if (status != 0) {
        throw "Could not read image header";
    }
}

void readFitsAttribute(fitsfile* filePtr, int i, std::string& name, std::string& value) {
    int status(0);
    char keyTmp[255];
    char valueTmp[255];
    
    fits_read_keyn(filePtr, i, keyTmp, valueTmp, NULL, &status);
    
    if (status != 0) {
        throw "Could not read attribute from header";
    }
    
    name = keyTmp;
    value = valueTmp;
}

void readFitsStringAttribute(fitsfile* filePtr, const std::string& name, std::string& value) {
    int status(0);
    int strLen;
    char strValueTmp[255];
    
    fits_read_string_key(filePtr, name.c_str(), 1, 255, strValueTmp, &strLen, NULL, &status);

    if (status != 0) {
        throw "Could not read string attribute";
    }
    
    value = strValueTmp;
}

// Reads size pixels starting at (row, channel, stokes) into destination, in the order of the
// standard dataset [stokes][channel][y][x], i.e. consecutive channels of one Stokes follow each other.
// In a FITS file with axis order STOKES,FREQ (bSwapStokesFreqAxis) consecutive planes in the file
// alternate between Stokes, so the read is split into one read per channel (each one contiguous).
static void readFitsPixels(fitsfile* filePtr, hsize_t channel, hsize_t row, unsigned int stokes, hsize_t size, float* destination, bool bSwapStokesFreqAxis, const char* errorMessage) {
    int status(0);

    if (!bSwapStokesFreqAxis) {
        long fpixel[] = {1, (long)row + 1, (long)channel + 1, stokes + 1};
        fits_read_pix(filePtr, TFLOAT, fpixel, size, NULL, destination, NULL, &status);
        if (status != 0) {
            throw errorMessage;
        }
        return;
    }

    long naxes[4] = {1, 1, 1, 1};
    fits_get_img_size(filePtr, 4, naxes, &status);
    if (status != 0) {
        throw errorMessage;
    }
    const hsize_t width = naxes[0];
    const hsize_t planeSize = naxes[0] * naxes[1];
    hsize_t offset = row * width; // position within the first plane, always at a row start

    while (size > 0) {
        hsize_t n = std::min(size, planeSize - offset);
        long fpixel[] = {1, (long)(offset / width) + 1, stokes + 1, (long)channel + 1};
        fits_read_pix(filePtr, TFLOAT, fpixel, n, NULL, destination, NULL, &status);
        if (status != 0) {
            throw errorMessage;
        }
        destination += n;
        size -= n;
        channel++;
        offset = 0;
    }
}

void readFitsData(fitsfile* filePtr, hsize_t channel, unsigned int stokes, hsize_t size, float* destination, bool bSwapStokesFreqAxis) {
    readFitsPixels(filePtr, channel, 0, stokes, size, destination, bSwapStokesFreqAxis, "Could not read image data");
}

void readFitsDataRow(fitsfile* filePtr, hsize_t channel, hsize_t row, unsigned int stokes, hsize_t size, float* destination, bool bSwapStokesFreqAxis) {
    readFitsPixels(filePtr, channel, row, stokes, size, destination, bSwapStokesFreqAxis, "Could not read image data row");
}

// Only available in C++ API from 1.10.1
bool hdf5Exists(H5::H5Location& location, const std::string& name) {
    return H5Lexists(location.getId(), name.c_str(), H5P_DEFAULT) > 0;
}

void createHdf5Dataset(H5::DataSet& dataset, H5::Group group, std::string path, H5::DataType dataType, std::vector<hsize_t> dims, const std::vector<hsize_t>& chunkDims) {
    auto splitPath = split(path, '/');
    
    auto name = splitPath.back();
    splitPath.pop_back();
    
    for (auto& groupname : splitPath) {
        if (!hdf5Exists(group, groupname)) {
            group = group.createGroup(groupname);
        } else {
            group = group.openGroup(groupname);
        }
    }
    
    H5::DSetCreatPropList propList;
    if (!chunkDims.empty()) {
        propList.setChunk(chunkDims.size(), chunkDims.data());
    }
    
    auto dataSpace = H5::DataSpace(dims.size(), dims.data());
    dataset = group.createDataSet(name, dataType, dataSpace, propList);
}

void writeHdf5Attribute(H5::Group group, std::string name, std::string value) {
    H5::StrType strType(H5::PredType::C_S1, 256);
    H5::DataSpace dataSpace(H5S_SCALAR);
    auto attribute = group.createAttribute(name, strType, dataSpace);
    attribute.write(strType, value);
}

void writeHdf5Attribute(H5::Group group, std::string name, int64_t value) {
    H5::IntType intType(H5::PredType::NATIVE_INT64);
    intType.setOrder(H5T_ORDER_LE);
    H5::DataSpace dataSpace(H5S_SCALAR);
    auto attribute = group.createAttribute(name, intType, dataSpace);
    attribute.write(intType, &value);
}

void writeHdf5Attribute(H5::Group group, std::string name, double value) {
    H5::FloatType doubleType(H5::PredType::NATIVE_DOUBLE);
    doubleType.setOrder(H5T_ORDER_LE);
    H5::DataSpace dataSpace(H5S_SCALAR);
    auto attribute = group.createAttribute(name, doubleType, dataSpace);
    attribute.write(doubleType, &value);
}

void writeHdf5Attribute(H5::Group group, std::string name, bool value) {
    H5::IntType boolType(H5::PredType::NATIVE_HBOOL);
    H5::DataSpace dataSpace(H5S_SCALAR);
    auto attribute = group.createAttribute(name, boolType, dataSpace);
    attribute.write(boolType, &value);
}

void writeHdf5Data(H5::DataSet& dataset, float* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    H5::DataSpace memSpace(dims.size(), dims.data());
    auto fileSpace = dataset.getSpace();
    if (!count.empty() && !start.empty()) {
        fileSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    }
    dataset.write(data, H5::PredType::NATIVE_FLOAT, memSpace, fileSpace);
}

void writeHdf5Data(H5::DataSet& dataset, double* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    H5::DataSpace memSpace(dims.size(), dims.data());
    auto fileSpace = dataset.getSpace();
    if (!count.empty() && !start.empty()) {
        fileSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    }
    dataset.write(data, H5::PredType::NATIVE_DOUBLE, memSpace, fileSpace);
}

void writeHdf5Data(H5::DataSet& dataset, int64_t* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    H5::DataSpace memSpace(dims.size(), dims.data());
    auto fileSpace = dataset.getSpace();
    if (!count.empty() && !start.empty()) {
        fileSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    }
    dataset.write(data, H5::PredType::NATIVE_INT64, memSpace, fileSpace);
}

void readHdf5Data(H5::DataSet& dataset, float* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count, const std::vector<hsize_t>& start) {
    H5::DataSpace memSpace(dims.size(), dims.data());
    auto fileSpace = dataset.getSpace();
    if (!count.empty() && !start.empty()) {
        fileSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    }
    dataset.read(data, H5::PredType::NATIVE_FLOAT, memSpace, fileSpace);
}

void RegionIndexToXYZ(hsize_t mipIndex, hsize_t& x, hsize_t& y, hsize_t& z, hsize_t width, hsize_t height, hsize_t factorX, hsize_t factorY, hsize_t factorZ) {
    int mipHeight = (height + factorY - 1)/ factorY;      //round result up
    int mipWidth = (width + factorX - 1) / factorX;       //round result up
    x = (mipIndex % (mipWidth * mipHeight)) % mipWidth * factorX;
    y = (mipIndex % (mipWidth * mipHeight)) / mipWidth * factorY;
    z = mipIndex / (mipWidth * mipHeight) * factorZ;
}

void getDividers(long int value, std::vector<int>& dividers) {
   dividers.clear();
   for(long int i=2;i<value;i++){
      if( (value % i) == 0 ) {
          dividers.push_back(i);
      }
   }
}

// false if the file is missing or does not start with a number (e.g. "max" = no limit in cgroup v2)
static bool readNumberFromFile(const std::string& path, hsize_t& value) {
    std::ifstream file(path);
    return bool(file >> value);
}

hsize_t detectMemoryLimit(std::string& source) {
    const hsize_t MiB = 1024 * 1024;
    const hsize_t NO_LIMIT = (hsize_t)1 << 60; // "unlimited" is a huge number in cgroup v1 (9223372036854771712)

    hsize_t limit = 0;
    source = "";
    auto consider = [&](hsize_t bytes, const std::string& name) {
        if (bytes == 0 || bytes >= NO_LIMIT) {
            return;
        }
        std::cout << "INFO : memory limit " << bytes * 1e-9 << " GB found in " << name << std::endl;
        if (limit == 0 || bytes < limit) {
            limit = bytes;
            source = name;
        }
    };

    // cgroup (enforced limit of a SLURM job). Lines of /proc/self/cgroup are HIERARCHY:CONTROLLERS:PATH,
    // with empty CONTROLLERS for cgroup v2. Limits of parent cgroups (SLURM: step, job, user) apply too.
    std::ifstream cgroupFile("/proc/self/cgroup");
    std::string line;
    while (std::getline(cgroupFile, line)) {
        const auto first = line.find(':');
        const auto second = line.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            continue;
        }
        const auto controllers = split(line.substr(first + 1, second - first - 1), ',');
        std::string path = line.substr(second + 1);

        std::string base, fileName;
        if (controllers.empty()) {
            base = "/sys/fs/cgroup";
            fileName = "memory.max";
        } else if (std::find(controllers.begin(), controllers.end(), "memory") != controllers.end()) {
            base = "/sys/fs/cgroup/memory";
            fileName = "memory.limit_in_bytes";
        } else {
            continue;
        }

        while (true) {
            const std::string file = base + path + "/" + fileName;
            hsize_t value = 0;
            if (readNumberFromFile(file, value)) {
                consider(value, file);
            }
            const auto slash = path.rfind('/');
            if (slash == std::string::npos) {
                break;
            }
            path = path.substr(0, slash);
        }
    }

    // SLURM request (when the cgroup files are not visible, e.g. in a container). Units are MiB, 0 = whole node
    const char* memPerNode = getenv("SLURM_MEM_PER_NODE");
    if (memPerNode) {
        consider(strtoull(memPerNode, nullptr, 10) * MiB, "SLURM_MEM_PER_NODE");
    }
    const char* memPerCpu = getenv("SLURM_MEM_PER_CPU");
    const char* cpusOnNode = getenv("SLURM_CPUS_ON_NODE");
    if (memPerCpu && cpusOnNode) {
        consider(strtoull(memPerCpu, nullptr, 10) * strtoull(cpusOnNode, nullptr, 10) * MiB, "SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE");
    }

    // ulimit -v
    struct rlimit addressSpace;
    if (getrlimit(RLIMIT_AS, &addressSpace) == 0 && addressSpace.rlim_cur != RLIM_INFINITY) {
        consider(addressSpace.rlim_cur, "ulimit -v (RLIMIT_AS)");
    }

    // memory free now, including reclaimable page cache
    std::ifstream memInfo("/proc/meminfo");
    while (std::getline(memInfo, line)) {
        hsize_t kB = 0;
        if (sscanf(line.c_str(), "MemAvailable: %llu kB", (unsigned long long*)&kB) == 1) {
            consider(kB * 1024, "MemAvailable in /proc/meminfo");
            break;
        }
    }

    return limit;
}
StatusFile::StatusFile(const std::string& _path) : path(_path), start(std::chrono::steady_clock::now()) {
    write("STARTED");
}

StatusFile::~StatusFile() {
    write(status);
}

void StatusFile::set(const std::string& _status, const std::string& _message) {
    status = _status;
    message = _message;
}

void StatusFile::add(const std::string& key, const std::string& value) {
    for (auto& field : fields) {
        if (field.first == key) {
            field.second = value;
            return;
        }
    }
    fields.push_back({key, value});
}

void StatusFile::write(const std::string& currentStatus) const {
    if (path.empty()) {
        return;
    }
    // must not throw: also called from the destructor
    const std::string tmpPath = path + ".tmp";
    {
        std::ofstream out(tmpPath);
        if (!out) {
            std::cerr << "WARNING : could not write status file " << tmpPath << std::endl;
            return;
        }
        std::string oneLineMessage = message;
        std::replace(oneLineMessage.begin(), oneLineMessage.end(), '\n', ' ');
        out << "STATUS=" << currentStatus << std::endl;
        if (!oneLineMessage.empty()) {
            out << "MESSAGE=" << oneLineMessage << std::endl;
        }
        for (auto& field : fields) {
            out << field.first << "=" << field.second << std::endl;
        }
        out << "ELAPSED_S=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << std::endl;
        out.close();
        if (out.fail()) {
            // e.g. disk full: keep the previous status file rather than replacing it with an empty / partial one
            std::cerr << "WARNING : could not write status file " << tmpPath << " -> " << path << " not updated" << std::endl;
            std::remove(tmpPath.c_str());
            return;
        }
    }
    if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
        std::cerr << "WARNING : could not rename " << tmpPath << " to " << path << std::endl;
    }
}
