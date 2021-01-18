/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include <getopt.h>
#include "Converter.h"

bool getOptions(int argc, char** argv, std::string& inputFileName, std::string& outputFileName, bool& slow, bool& progress, bool& onlyReportMemory) {
    extern int optind;
    extern char *optarg;
    
    int opt;
    bool err(false);
    
    std::ostringstream usage;
    usage << "IDIA FITS to HDF5 converter version " << HDF5_CONVERTER_VERSION 
    << " using IDIA schema version " << SCHEMA_VERSION << std::endl
    << "Usage: hdf_convert [-o output_filename] [-s] [-p] [-m] input_filename" << std::endl << std::endl
    << "Options:" << std::endl 
    << "-o\tOutput filename" << std::endl 
    << "-s\tUse slower but less memory-intensive method (enable if memory allocation fails)" << std::endl 
    << "-p\tPrint progress output (by default the program is silent)" << std::endl
    << "-m\tReport predicted memory usage and exit without performing the conversion" << std::endl
    << "-q\tSuppress all non-error output. Deprecated; this is now the default." << std::endl;
    
    while ((opt = getopt(argc, argv, ":o:spqm")) != -1) {
        switch (opt) {
            case 'o':
                outputFileName.assign(optarg);
                break;
            case 's':
                // use slower but less memory-intensive method
                slow = true;
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
    bool progress(false);
    bool onlyReportMemory(false);
    
    if (!getOptions(argc, argv, inputFileName, outputFileName, slow, progress, onlyReportMemory)) {
        return 1;
    }
    
    std::unique_ptr<Converter> converter;
        
    try {
        converter = Converter::getConverter(inputFileName, outputFileName, slow, progress);
        
        if (onlyReportMemory) {
            converter->reportMemoryUsage();
            return 0;
        }
    
        DEBUG(std::cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << std::endl;);

        converter->convert();
    } catch (const char* msg) {
        std::cerr << "Error: " << msg << ". Aborting." << std::endl;
        return 1;
    }

    return 0;
}
