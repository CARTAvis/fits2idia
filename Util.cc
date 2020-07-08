#include "Util.h"

std::vector<std::string> split(const std::string &str, char separator) {
    std::vector<std::string> result;
    std::istringstream stream(str);
    std::string substr;
    while (std::getline(stream, substr, separator)) {
        result.push_back(substr);
    }
    return result;
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

void readFitsData(fitsfile* filePtr, hsize_t channel, unsigned int stokes, hsize_t size, float* destination) {
    long fpixel[] = {1, 1, (long)channel + 1, stokes + 1};
    int status(0);
    
    fits_read_pix(filePtr, TFLOAT, fpixel, size, NULL, destination, NULL, &status);
    
    if (status != 0) {
        throw "Could not read image data";
    }
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
