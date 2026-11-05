/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// idia2fits : back-converter writing the standard dataset (/<hdu>/DATA) of an IDIA HDF5 file
// and the FITS header stored as attributes of /<hdu> to a FITS image.
//
// What fits2idia stored and how it is restored here:
//   - DATA is float32 with FITS axis order reversed (C order: [stokes,] [freq,] y, x)
//     -> written as BITPIX = -32 with NAXISn taken from the DATA shape. Pixels were read with
//        TFLOAT, i.e. BSCALE/BZERO were already applied -> those keywords (and BLANK) are not copied.
//   - header keywords are typed attributes (string, int64, double, bool)
//     -> mandatory keywords come first, the remaining ones in the order HDF5 returns them.
//   - COMMENT and HISTORY cards were never stored by fits2idia -> they cannot be restored.
//     Long strings are stored complete, so the CONTINUE attribute (always empty) is skipped.
//   - if fits2idia swapped STOKES/FREQ axes, DATA and the stored CTYPE/CRVAL/... are already
//     swapped consistently -> the output has FREQ as axis 3 and STOKES as axis 4.

#include <getopt.h>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#include <H5Cpp.h>
#include <fitsio.h>

// read buffer size without -M:
static const hsize_t DEFAULT_BUFFER_BYTES = 256 * 1024 * 1024;
// part of the -M limit kept for the process itself, HDF5 (incl. its 1 MB chunk cache) and cfitsio buffers:
static const hsize_t MEMORY_OVERHEAD_BYTES = 32 * 1000 * 1000;

struct Options {
    std::string inputFileName;
    std::string outputFileName;
    std::string hdu{"0"};
    bool overwrite{false};
    bool progress{false};
    int memoryLimitInMb{0}; // 0 = not set -> DEFAULT_BUFFER_BYTES
};

void usage(const char* name) {
    std::cerr << "Usage: " << name << " [-o output.fits] [-H hdu] [-M memory_limit_MB] [-f] [-p] input.hdf5" << std::endl
              << "-o\tOutput FITS file [default: input name with .hdf5/.h5 replaced by .fits]" << std::endl
              << "-H\tHDU group in the HDF5 file [default: 0]" << std::endl
              << "-M\tMemory limit in MB, sets the read buffer size [default: buffer of "
              << DEFAULT_BUFFER_BYTES / (1024 * 1024) << " MiB]" << std::endl
              << "-f\tOverwrite the output file if it exists" << std::endl
              << "-p\tPrint progress output" << std::endl;
}

bool getOptions(int argc, char** argv, Options& options) {
    int opt;
    while ((opt = getopt(argc, argv, ":o:H:M:fp")) != -1) {
        switch (opt) {
            case 'o': options.outputFileName = optarg; break;
            case 'H': options.hdu = optarg; break;
            case 'M':
                options.memoryLimitInMb = atoi(optarg);
                if (options.memoryLimitInMb <= 0) {
                    std::cerr << "ERROR : -M needs a positive memory limit in MB" << std::endl;
                    return false;
                }
                break;
            case 'f': options.overwrite = true; break;
            case 'p': options.progress = true; break;
            default:
                usage(argv[0]);
                return false;
        }
    }
    if (optind != argc - 1) {
        usage(argv[0]);
        return false;
    }
    options.inputFileName = argv[optind];

    if (options.outputFileName.empty()) {
        std::string base = options.inputFileName;
        for (const std::string ext : {".hdf5", ".h5"}) {
            if (base.size() > ext.size() && base.compare(base.size() - ext.size(), ext.size(), ext) == 0) {
                base.erase(base.size() - ext.size());
                break;
            }
        }
        options.outputFileName = base + ".fits";
    }
    return true;
}

void checkFitsStatus(int status, const std::string& what) {
    if (status != 0) {
        char message[FLEN_STATUS];
        fits_get_errstatus(status, message);
        std::cerr << "ERROR : " << what << " : " << message << std::endl;
        throw "FITS error";
    }
}

// keywords which describe the original FITS data layout or were added by fits2idia:
bool skipKeyword(const std::string& name) {
    static const std::set<std::string> skipped = {
        "SIMPLE", "BITPIX", "NAXIS", "EXTEND", "END", "BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM",
        "CONTINUE", "COMMENT", "HISTORY",
        "SCHEMA_VERSION", "HDF5_CONVERTER", "HDF5_CONVERTER_VERSION", "HDF5_CONVERTER_TYPE"
    };
    if (skipped.count(name)) {
        return true;
    }
    if (name.rfind("HDF5_", 0) == 0) { // e.g. HDF5_PARAM_CUBE_HISTOGRAM_APPROXIMATE
        return true;
    }
    if (name.size() > 5 && name.rfind("NAXIS", 0) == 0 && std::all_of(name.begin() + 5, name.end(), ::isdigit)) {
        return true;
    }
    return false;
}

std::string readStringAttribute(H5::Attribute& attribute) {
    std::string value;
    attribute.read(attribute.getStrType(), value);
    auto end = value.find('\0');
    if (end != std::string::npos) {
        value.erase(end);
    }
    return value;
}

// copies one attribute to the FITS header, returns false if its type is not supported:
bool writeKeyword(fitsfile* fptr, H5::Attribute& attribute, const std::string& name) {
    int status(0);
    auto typeClass = attribute.getTypeClass();

    if (typeClass == H5T_STRING) {
        std::string value = readStringAttribute(attribute);
        // fits2idia stores the raw FITS value (e.g. 'FREQ    ' with quotes) for keywords renamed by the
        // STOKES <-> FREQ axis swap -> strip the quotes and padding so they are not quoted twice
        if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
            value = value.substr(1, value.size() - 2);
            value.erase(value.find_last_not_of(' ') + 1);
            for (auto pos = value.find("''"); pos != std::string::npos; pos = value.find("''", pos + 1)) {
                value.erase(pos, 1);
            }
        }
        if (value.size() > 68) {
            fits_write_key_longstr(fptr, name.c_str(), value.c_str(), NULL, &status);
        } else {
            fits_update_key_str(fptr, name.c_str(), value.c_str(), NULL, &status);
        }
    } else if (typeClass == H5T_FLOAT) {
        double value;
        attribute.read(H5::PredType::NATIVE_DOUBLE, &value);
        fits_update_key_dbl(fptr, name.c_str(), value, -15, NULL, &status);
    } else if (typeClass == H5T_INTEGER && attribute.getIntType().getSize() == 1) {
        // fits2idia stores FITS logicals as NATIVE_HBOOL (1 byte); no 1-byte integers are stored otherwise
        unsigned char value;
        attribute.read(H5::PredType::NATIVE_UCHAR, &value);
        int logical = value ? 1 : 0;
        fits_update_key_log(fptr, name.c_str(), logical, NULL, &status);
    } else if (typeClass == H5T_INTEGER) {
        LONGLONG value;
        attribute.read(H5::PredType::NATIVE_LLONG, &value);
        fits_update_key_lng(fptr, name.c_str(), value, NULL, &status);
    } else {
        return false;
    }
    checkFitsStatus(status, "writing keyword " + name);
    return true;
}

void copyHeader(fitsfile* fptr, H5::Group& group) {
    int numAttributes = group.getNumAttrs();
    int nCopied(0), nSkipped(0);

    for (int i = 0; i < numAttributes; i++) {
        H5::Attribute attribute = group.openAttribute((unsigned int)i);
        std::string name = attribute.getName();

        if (skipKeyword(name)) {
            nSkipped++;
            continue;
        }
        if (attribute.getSpace().getSimpleExtentNpoints() != 1) {
            std::cout << "WARNING : skipping non-scalar attribute " << name << std::endl;
            nSkipped++;
            continue;
        }
        if (!writeKeyword(fptr, attribute, name)) {
            std::cout << "WARNING : skipping attribute " << name << " of unsupported type" << std::endl;
            nSkipped++;
            continue;
        }
        nCopied++;
    }
    std::cout << "INFO : copied " << nCopied << " header keywords, skipped " << nSkipped << std::endl;
}

void addHistory(fitsfile* fptr, H5::Group& group, const Options& options) {
    int status(0);
    std::string history = "idia2fits: converted from " + options.inputFileName;
    if (group.attrExists("HDF5_CONVERTER") && group.attrExists("HDF5_CONVERTER_VERSION")) {
        H5::Attribute converter = group.openAttribute("HDF5_CONVERTER");
        H5::Attribute version = group.openAttribute("HDF5_CONVERTER_VERSION");
        history += " (written by " + readStringAttribute(converter) + " " + readStringAttribute(version) + ")";
    }
    fits_write_history(fptr, history.c_str(), &status);
    fits_write_history(fptr, "idia2fits: COMMENT/HISTORY cards of the original file are not stored in IDIA HDF5", &status);
    checkFitsStatus(status, "writing HISTORY");
}

// Reads DATA in blocks of at most bufferBytes and appends them to the FITS image (FITS and DATA share
// the same memory order). Whole planes fit -> several consecutive channels per read (fast for small
// planes); otherwise bands of rows aligned to the chunk height, so that every chunk is read only once.
// bufferBytes too small for one band: error when it comes from -M (limitGiven), else one band is used.
void copyData(fitsfile* fptr, H5::DataSet& dataset, const std::vector<hsize_t>& dims, hsize_t bufferBytes, bool limitGiven, bool progress) {
    const int N = dims.size();
    const hsize_t width = dims[N - 1];
    const hsize_t height = dims[N - 2];
    const hsize_t depth = N >= 3 ? dims[N - 3] : 1;
    const hsize_t stokes = N == 4 ? dims[0] : 1;
    const hsize_t planeValues = height * width;

    hsize_t chunkRows(1);
    H5::DSetCreatPropList propList = dataset.getCreatePlist();
    if (propList.getLayout() == H5D_CHUNKED) {
        std::vector<hsize_t> chunkDims(N);
        propList.getChunk(N, chunkDims.data());
        chunkRows = std::min(chunkDims[N - 2], height);
    }

    const hsize_t bufferValues = bufferBytes / sizeof(float);
    const hsize_t minValues = chunkRows * width;
    if (bufferValues < minValues && limitGiven) {
        hsize_t minimumMb = (minValues * sizeof(float) + MEMORY_OVERHEAD_BYTES + 999999) / 1000000;
        std::cerr << "ERROR : one band of " << chunkRows << " rows needs " << minValues * sizeof(float) * 1e-6
                  << " MB of buffer -> use -M " << minimumMb << " or more" << std::endl;
        throw "Memory limit too small";
    }

    hsize_t planesPerBlock(1), rowsPerBlock(height);
    if (bufferValues >= planeValues) {
        planesPerBlock = std::min(depth, bufferValues / planeValues);
    } else {
        rowsPerBlock = std::max(chunkRows, (bufferValues / width / chunkRows) * chunkRows);
        rowsPerBlock = std::min(rowsPerBlock, height);
    }
    std::vector<float> buffer(planesPerBlock * rowsPerBlock * width);
    std::cout << "INFO : read buffer = " << buffer.size() * sizeof(float) * 1e-6 << " MB ("
              << planesPerBlock << " plane(s) x " << rowsPerBlock << " rows per read)" << std::endl;

    H5::DataSpace fileSpace = dataset.getSpace();
    const hsize_t numPlanes = stokes * depth;
    hsize_t planesDone(0), nextProgress(0);

    for (hsize_t s = 0; s < stokes; s++) {
        for (hsize_t z = 0; z < depth; z += planesPerBlock) {
            const hsize_t planes = std::min(planesPerBlock, depth - z);

            for (hsize_t y = 0; y < height; y += rowsPerBlock) {
                const hsize_t rows = std::min(rowsPerBlock, height - y);

                // C order of DATA is [stokes,] [freq,] y, x
                std::vector<hsize_t> start(N, 0), count(N, 1);
                start[N - 2] = y;
                count[N - 2] = rows;
                count[N - 1] = width;
                if (N >= 3) {
                    start[N - 3] = z;
                    count[N - 3] = planes;
                }
                if (N == 4) {
                    start[0] = s;
                }

                const hsize_t numValues = planes * rows * width;
                fileSpace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
                hsize_t memDims[] = {numValues};
                H5::DataSpace memSpace(1, memDims);
                dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT, memSpace, fileSpace);

                // FITS first pixel is 1-based and in reversed axis order; the block is contiguous in
                // the FITS file because several planes are only read when they are read whole
                std::vector<long> fpixel(N);
                for (int i = 0; i < N; i++) {
                    fpixel[i] = start[N - 1 - i] + 1;
                }
                int status(0);
                fits_write_pix(fptr, TFLOAT, fpixel.data(), (LONGLONG)numValues, buffer.data(), &status);
                checkFitsStatus(status, "writing image data");
            }

            planesDone += planes;
            if (progress && (planesDone >= nextProgress || planesDone == numPlanes)) {
                std::cout << "PROGRESS : plane " << planesDone << " / " << numPlanes << std::endl;
                nextProgress = planesDone + std::max((hsize_t)1, numPlanes / 20);
            }
        }
    }
}

int main(int argc, char** argv) {
    Options options;
    if (!getOptions(argc, argv, options)) {
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    fitsfile* fptr = NULL;

    try {
        H5::Exception::dontPrint();
        H5::H5File inputFile(options.inputFileName, H5F_ACC_RDONLY);
        H5::Group group = inputFile.openGroup(options.hdu);
        H5::DataSet dataset = group.openDataSet("DATA");

        H5::DataSpace space = dataset.getSpace();
        const int N = space.getSimpleExtentNdims();
        if (N < 2 || N > 4) {
            std::cerr << "ERROR : DATA has " << N << " dimensions, 2 - 4 are supported" << std::endl;
            return 1;
        }
        std::vector<hsize_t> dims(N);
        space.getSimpleExtentDims(dims.data());

        std::vector<long> naxes(N);
        std::cout << "INFO : DATA dimensions (FITS order) = ";
        for (int i = 0; i < N; i++) {
            naxes[i] = dims[N - 1 - i];
            std::cout << naxes[i] << (i < N - 1 ? " x " : "\n");
        }

        int status(0);
        std::string fitsName = (options.overwrite ? "!" : "") + options.outputFileName;
        fits_create_file(&fptr, fitsName.c_str(), &status);
        checkFitsStatus(status, "creating " + options.outputFileName + (options.overwrite ? "" : " (use -f to overwrite)"));
        fits_create_img(fptr, FLOAT_IMG, N, naxes.data(), &status);
        checkFitsStatus(status, "creating primary image");

        copyHeader(fptr, group);
        addHistory(fptr, group, options);
        hsize_t bufferBytes = DEFAULT_BUFFER_BYTES;
        if (options.memoryLimitInMb > 0) {
            hsize_t memoryLimit = (hsize_t)options.memoryLimitInMb * 1000000; // same MB convention as fits2idia
            bufferBytes = memoryLimit > MEMORY_OVERHEAD_BYTES ? memoryLimit - MEMORY_OVERHEAD_BYTES : 0;
            std::cout << "INFO : memory limit = " << options.memoryLimitInMb << " MB" << std::endl;
        }
        copyData(fptr, dataset, dims, bufferBytes, options.memoryLimitInMb > 0, options.progress);

        fits_close_file(fptr, &status);
        fptr = NULL;
        checkFitsStatus(status, "closing " + options.outputFileName);
    } catch (const H5::Exception& e) {
        std::cerr << "ERROR : HDF5 : " << e.getFuncName() << " : " << e.getDetailMsg() << std::endl;
        if (fptr) {
            int status(0);
            fits_delete_file(fptr, &status); // do not leave a partial FITS file behind
        }
        return 1;
    } catch (const char* msg) {
        if (fptr) {
            int status(0);
            fits_delete_file(fptr, &status);
        }
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "INFO : written " << options.outputFileName << " in "
              << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
    return 0;
}
