/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include <getopt.h>
#include <regex>
#include <fstream>
#include <sstream>
#include "Converter.h"

bool getOptions(int argc, char** argv, std::string& inputFileName, std::string& outputFileName, bool& slow, bool& smart, bool& progress, bool& onlyReportMemory, bool& zMips, int& memoryLimitInMb, bool& auto_mode, int& n_io_blocks) {
    extern int optind;
    extern char *optarg;
    
    int opt;
    bool err(false);
    
    std::ostringstream usage;
    usage << "IDIA FITS to HDF5 converter version " << HDF5_CONVERTER_VERSION 
    << " using IDIA schema version " << SCHEMA_VERSION << std::endl
    << "Usage: fits2idia [-o output_filename] [-s] [-p] [-m] [-z] input_filename" << std::endl << std::endl
    << "Options:" << std::endl 
    << "-a\tUse auto mode adjusting memory usage below the limit" << std::endl
    << "-B\tNumber of freq-images (i.e. blocks) read in one read transation [default " << n_io_blocks << "]" << std::endl
    << "-o\tOutput filename" << std::endl 
    << "-s\tUse slower but less memory-intensive method (enable if memory allocation fails)" << std::endl 
    << "-S\tUse smart converter with MPI optimisations and still using small amount of memory (use -a to automatically adjust)" << std::endl 
    << "-p\tPrint progress output (by default the program is silent)" << std::endl    
    << "-r\tUse auto mode adjusting memory usage below the limit (only for backward compatibility with the previous version of smart converter)" << std::endl
    << "-m\tReport predicted memory usage and exit without performing the conversion" << std::endl
    << "-M\tSpecify memory limit in MB" << std::endl
    << "-q\tSuppress all non-error output. Deprecated; this is now the default." << std::endl
    << "-z\tInclude axis 3 in mipmap calculation (currently not compatible with -s mode)." << std::endl;

    while ((opt = getopt(argc, argv, ":o:arsSpqmzM:B:")) != -1) {
        switch (opt) {
            case 'a':
                auto_mode = true;
                break;
            case 'B':
                if (optarg) {
                   n_io_blocks = atol(optarg);
                }
                break;
            case 'r':
                auto_mode = true;
                break;
            case 'o':
                outputFileName.assign(optarg);
                break;
            case 's':
                // use slower but less memory-intensive method
                slow = true;
                break;
            case 'S':
                // use smart converter
                smart = true;
                break;
            case 'p':
                progress = true;
                break;
            case 'q':
                std::cerr << "The -q flag is deprecated. The converter is quiet by default." << std::endl;
                break;
            case 'm':
                // only print memory usage and exit
                onlyReportMemory = true;
                break;
            case 'M':
                if (optarg) {
                    memoryLimitInMb = atof(optarg);   
                }
                break;
            case 'z':
                zMips = true;
                break;
            case ':':
                err = true;
                std::cerr << "Missing argument for option " << opt << "." << std::endl;
                break;
            case '?':
                err = true;
                std::cerr << "Unknown option " << opt << "." << std::endl;
                break;
        }
    }
    
    if (optind >= argc) {
        err = true;
        std::cerr << "Missing input filename parameter." << std::endl;
    } else {
        inputFileName.assign(argv[optind]);
        optind++;
    }
    
    if (argc > optind) {
        err = true;
        std::cerr << "Unexpected additional parameters." << std::endl;
    }
        
    if (err) {
        std::cerr << std::endl << usage.str() << std::endl;
        return false;
    }
    
    if (outputFileName.empty()) {
        auto fitsIndex = inputFileName.find_last_of(".fits");
        if (fitsIndex != std::string::npos) {
            outputFileName = inputFileName.substr(0, fitsIndex - 4);
            outputFileName += ".hdf5";
        } else {
            outputFileName = inputFileName + ".hdf5";
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    std::string inputFileName;
    std::string outputFileName;
    bool slow(false);
    bool smart(false);
    bool auto_mode(false);
    bool progress(false);
    bool onlyReportMemory(false);
    bool zMips(false);
    int memoryLimitInMb(0);
    int n_io_blocks(1);
    
    if (!getOptions(argc, argv, inputFileName, outputFileName, slow, smart,  progress, onlyReportMemory, zMips, memoryLimitInMb, auto_mode, n_io_blocks)) {
        return 1;
    }

    if (slow && zMips){
        std::cerr << "Currently unable to include depth in mipmap calculation for -s mode." << std::endl;
        return -1;
    }

    hsize_t memoryLimit(0);
    
    std::ifstream rcFile("/etc/fits2idiarc");
    if (rcFile.fail()){
        DEBUG(std::cout << "No system configuration file found." << std::endl;);
    } else {
        std::string line;
        while (std::getline(rcFile, line)){
            if (std::regex_match(line, std::regex("#.*"))) {
                continue;
            }
            std::smatch match;
            if (std::regex_match(line, match, std::regex(" *memory_limit *= *(\\d+) *"))) {
                std::stringstream sstream(match[1]);
                sstream >> memoryLimit;
            }
        }
    }
    if (memoryLimitInMb > 0) {
       memoryLimit = memoryLimitInMb * 1e6; // converting from MB to bytes
    }
    
    std::unique_ptr<Converter> converter;
        
    try {
        converter = Converter::getConverter(inputFileName, outputFileName, slow, smart, progress, zMips, memoryLimitInMb);
        
        if (n_io_blocks>1) {
           converter->setIOBlocks(n_io_blocks);
        }
        
        if (onlyReportMemory) {
            converter->reportMemoryUsage();
            return 0;
        }
        
        if (memoryLimit > 0) {
            hsize_t predictedTotal = converter->calculateMemoryUsage().total;
            if (predictedTotal > memoryLimit) {
                bool ok = false;
                if (auto_mode) {
                    std::cout << "Required predicted memory " << predictedTotal * 1e-9 << "GB exceeds memory limit of " << memoryLimit * 1e-9 << "GB." << std::endl;
                    std::cout << "This is automatic mode -> trying to reduce the required memory limit to continue processing" << std::endl;
                    // std::cout << "WARNING : this is not fully implemented yet -> exiting now!" << std::endl;
                    
                    if( converter->ReduceMemoryUsage( memoryLimit, 10 ) ) {
                       predictedTotal = converter->calculateMemoryUsage().total;
                       std::cout << "SUCCESS : reduced memory usage to " << predictedTotal * 1e-9 << "GB whcich is below the limit of " << memoryLimit * 1e-9 << "GB." << std::endl;
                       ok = true;
                    }
                } else {
                   std::cerr << "Error: approximate memory requirement of " << predictedTotal * 1e-9 << "GB exceeds configured memory limit of " << memoryLimit * 1e-9 << "GB. Aborting." << std::endl;
                   std::cerr << "Suggestion: try using -a option to automatically reduce the required memory usage." << std::endl;
                } 
                
                if( !ok ) {
                    std::cerr << "Error: approximate memory requirement of " << predictedTotal * 1e-9 << "GB exceeds configured memory limit of " << memoryLimit * 1e-9 << "GB. Aborting." << std::endl;
                    return 1;
                }
            }
        }
    
        DEBUG(std::cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << std::endl;);

        converter->convert();
    } catch (const char* msg) {
        std::cerr << "Error: " << msg << ". Aborting." << std::endl;
        return 1;
    }

    return 0;
}
