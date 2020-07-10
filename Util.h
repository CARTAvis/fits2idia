#ifndef __UTIL_H
#define __UTIL_H

#include "common.h"

std::vector<std::string> split(const std::string &str, char separator);
std::vector<hsize_t> trimAxes(const std::vector<hsize_t>& dims, int N);
std::vector<hsize_t> extend(const std::vector<hsize_t>& left, const std::vector<hsize_t>& right);
hsize_t product(const std::vector<hsize_t>& dims);
bool useChunks(hsize_t width, hsize_t height);

void openFitsFile(fitsfile** filePtrPtr, const std::string& fileName);
void getFitsDims(fitsfile* filePtr, int& N, long* dims);
void readFitsHeader(fitsfile* filePtr, int& numAttributes);
void readFitsAttribute(fitsfile* filePtr, int i, std::string& name, std::string& value);
void readFitsStringAttribute(fitsfile* filePtr, const std::string& name, std::string& value);
void readFitsData(fitsfile* filePtr, hsize_t channel, unsigned int stokes, hsize_t size, float* destination);

// Only available in C++ API from 1.10.1
bool hdf5Exists(H5::H5Location& location, const std::string& name);
void createHdf5Dataset(H5::DataSet& dataset, H5::Group group, std::string path, H5::DataType dataType, std::vector<hsize_t> dims, const std::vector<hsize_t>& chunkDims = EMPTY_DIMS);
void writeHdf5Attribute(H5::Group group, std::string name, std::string value);
void writeHdf5Attribute(H5::Group group, std::string name, int64_t value);
void writeHdf5Attribute(H5::Group group, std::string name, double value);
void writeHdf5Attribute(H5::Group group, std::string name, bool value);

void writeHdf5Data(H5::DataSet& dataset, float* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);
void writeHdf5Data(H5::DataSet& dataset, double* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);
void writeHdf5Data(H5::DataSet& dataset, int64_t* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);

void readHdf5Data(H5::DataSet& dataset, float* data, const std::vector<hsize_t>& dims, const std::vector<hsize_t>& count = EMPTY_DIMS, const std::vector<hsize_t>& start = EMPTY_DIMS);

#endif
