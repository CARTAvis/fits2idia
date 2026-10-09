/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include <getopt.h>
#include <regex>
#include <fstream>
#include <sstream>
#include "Converter.h"

// gethostname:
#include <unistd.h>
#include <climits>   // HOST_NAME_MAX
#include <string>

eSmartConverterType parse_smartconverter_type(const char* smartconverter_type) {
   char first_char[2];
   first_char[0] = smartconverter_type[0];
   first_char[1] = '\0';
   
   printf("DEBUG : parse_smartconverter_type(%s)\n",smartconverter_type);

   if (strcasecmp(smartconverter_type,"micro")==0 || strcasecmp(first_char,"m")==0) {
      return eSmartMicroMemoryConverter;
   }
   if (strcasecmp(smartconverter_type,"spatial")==0 || strcasecmp(first_char,"s")==0) {
      return eSmartConverterSpatialParallel;
   }
   if (strcasecmp(smartconverter_type,"frequency")==0 || strcasecmp(smartconverter_type,"channel")==0 || strcasecmp(smartconverter_type,"fast")==0 ||
       strcasecmp(first_char,"f")==0 || strcasecmp(first_char,"c")==0) {
      return eSmartConverterChannelParallel;
   }
   if (strcasecmp(smartconverter_type,"frequency-twopass")==0 || strcasecmp(smartconverter_type,"channel-twopass")==0 || strcasecmp(smartconverter_type,"fast-twopass")==0 ||
       strcasecmp(smartconverter_type,"twopass")==0 || strcasecmp(smartconverter_type,"2pass")==0) {
      return eSmartConverterChannelParallelTwoPass;
   }
   return eAutoSelectedSmartConverter;
}

void getSystemName(std::string& systemName)
{
   // use system name from OS functions:
   char buf[HOST_NAME_MAX + 1] = {0};
   if (gethostname(buf, sizeof(buf)) != 0) {
      systemName  = "unknown-host";
   } else {
      systemName = buf;
   }
}

struct commandLineOptions 
{
   std::string inputFileName;
   std::string outputFileName;
   bool slow{false};
   bool smart{false};
   eSmartConverterType smartconverter_type{eAutoSelectedSmartConverter};
   bool progress;
   bool onlyReportMemory{false};
   bool onlyReportMemoryAndExectime{false};
   bool zMips{false};
   bool rotatedDatasetChunking{false};
   bool rotatedDatasetChunkingForced{false};   
   double maxSwizzledChunkMb{0}; // 0 = default
   std::string rotatedChunkDims; // -K w,h[,d], empty = automatic
   int memoryLimitInMb{0};
   bool auto_mode{false};
   int n_io_blocks{1};
   std::string exclude_list;
   std::string include_list;
   std::string systemName;
};

bool getOptions(int argc, char** argv, commandLineOptions& cmdLineOptions) {
    extern int optind;
    extern char *optarg;
    std::string converter_type="unknown";
    
    int opt;
    bool err(false);
    
    std::ostringstream usage;
    usage << "IDIA FITS to HDF5 converter version " << HDF5_CONVERTER_VERSION 
    << " using IDIA schema version " << SCHEMA_VERSION << std::endl
    << "Usage: fits2idia [-o output_filename] [-s] [-p] [-m] [-z] input_filename" << std::endl << std::endl
    << "Options:" << std::endl 
    << "-a\tUse auto mode adjusting memory usage below the limit" << std::endl
    << "-B\tNumber of freq-images (i.e. blocks) read in one read transation [default " << cmdLineOptions.n_io_blocks << "]" << std::endl
    << "-C\tChunk the rotated dataset if it is worth doing so [default: " << cmdLineOptions.rotatedDatasetChunking << "]" << std::endl
    << "-F\tForce chunking the rotated dataset regardless if worth doing but only in SLOW/SMART converters [default: " << cmdLineOptions.rotatedDatasetChunkingForced << "]" << std::endl
    << "-L\tMaximum chunk size of the rotated dataset in MB, used with -C/-F (SLOW / SMART converters only) [default: " << Converter::maxSwizzledChunkBytes / 1e6 << " MB]" << std::endl
    << "-K\tExplicit chunk dims of the rotated dataset as w,h[,d] (d defaults to full depth), implies -C, overrides -F/-L. For experiments (SLOW / SMART converters only)" << std::endl
    << "-o\tOutput filename" << std::endl 
    << "-s\tUse slower but less memory-intensive method (enable if memory allocation fails)" << std::endl 
    << "-S\tUse smart converter with MPI optimisations and still using small amount of memory (use -a to automatically adjust)" << std::endl 
    << "-T\tType of smart converter: 'spatial' paralellised over pixels [DEFAULT], 'frequency', 'channel', 'twopass', 'micro' or 'fast' (parallelised over channels)" << std::endl
    << "-p\tPrint progress output (by default the program is silent)" << std::endl    
    << "-r\tUse auto mode adjusting memory usage below the limit (only for backward compatibility with the previous version of smart converter)" << std::endl
    << "-H\tSystem name which can be used to use system specific I/O measurements (e.g. -H setonix), possible values: setonix, setonix-ssd, laptop" << std::endl
    << "-R\tReport predicted memory usage and execution time without performing the conversion" << std::endl
    << "-m\tReport predicted memory usage and exit without performing the conversion. This is for backward compatibility, use -R to also see predicted exection time." << std::endl
    << "-M\tSpecify memory limit in MB" << std::endl
    << "-q\tSuppress all non-error output. Deprecated; this is now the default." << std::endl
    << "-z\tInclude axis 3 in mipmap calculation (currently not compatible with -s mode)." << std::endl
    << std::endl
    << "Additional options specifying parameters of the convertion (e.g. allow for approximations):" << std::endl
    << "-A\tUse approximated calculation of cube (XYZ) histogram using channel histograms [default " << SmartConverter::bApproximateCubeHistogram << " ]" << std::endl
    << "-E\tExclude specific datasets which can be: r (rotated), s (standard), m (mipmaps), h (channel histograms), c (cube histogram)" << std::endl
    << "-I\tIxclude specific datasets which can be: r (rotated), s (standard), m (mipmaps), h (channel histograms), c (cube histogram)" << std::endl;

    while ((opt = getopt(argc, argv, ":o:arsSpqmRCFzM:B:AT:H:E:I:K:L:")) != -1) {
        switch (opt) {
            case 'a':
            case 'r':
                cmdLineOptions.auto_mode = true;
                cmdLineOptions.n_io_blocks = -1; // it will be automatically calculated based on memory limit
                break;
            case 'A':
                SmartConverter::bApproximateCubeHistogram = true;
                break;
            case 'B':
                if (optarg) {
                   cmdLineOptions.n_io_blocks = atol(optarg);
                }
                break;
            case 'C':
                cmdLineOptions.rotatedDatasetChunking = true;
                Converter::rotatedDatasetChunking = true;
                break;

            case 'F':
                cmdLineOptions.rotatedDatasetChunking = true;
                cmdLineOptions.rotatedDatasetChunkingForced = true;
                Converter::rotatedDatasetChunking = true;      // -F implies -C
                Converter::rotatedDatasetChunkingForce = true;
                break;

            case 'E':
                if (optarg) {
                   cmdLineOptions.exclude_list = optarg;
                }
                break;
            case 'H':
                if (optarg) {
                   cmdLineOptions.systemName = optarg;
                }
                break;
            case 'I':
                if (optarg) {
                   cmdLineOptions.include_list = optarg;
                }
                break;

            case 'K':
                if (optarg) {
                    std::vector<hsize_t> dims;
                    bool bad = false;
                    for (auto& tok : split(optarg, ',')) {
                        char* end = nullptr;
                        unsigned long long v = strtoull(tok.c_str(), &end, 10);
                        if (tok.empty() || *end != '\0' || v == 0) { bad = true; break; }
                        dims.push_back((hsize_t)v);
                    }
                    if (bad || dims.size() < 2 || dims.size() > 3) {
                        std::cerr << "-K expects w,h or w,h,d (all > 0), e.g. -K 64,64 or -K 64,64,3842" << std::endl;
                        err = true;
                    } else {
                        cmdLineOptions.rotatedChunkDims = optarg;
                        cmdLineOptions.rotatedDatasetChunking = true;
                        Converter::rotatedChunkOverride = dims;
                        Converter::rotatedDatasetChunking = true;   // -K implies -C
                    }
                }
                break;                
            case 'L':
                if (optarg) {
                    double mb = atof(optarg);
                    if (mb <= 0 || mb >= 4096) {   // HDF5 chunks must be < 4 GB
                        std::cerr << "-L expects a chunk size limit in MB, 0 < L < 4096" << std::endl;
                        err = true;
                    } else {
                        Converter::maxSwizzledChunkBytes = (hsize_t)(mb * 1e6);
                        cmdLineOptions.maxSwizzledChunkMb = mb;
                    }
                }
                break;
                                
/*            case 'r':
                cmdLineOptions.auto_mode = true;
                break;*/
            case 'o':
                cmdLineOptions.outputFileName.assign(optarg);
                break;
            case 's':
                // use slower but less memory-intensive method
                cmdLineOptions.slow = true;
                break;
            case 'S':
                // use smart converter
                cmdLineOptions.smart = true;
                break;
            case 'p':
                cmdLineOptions.progress = true;
                break;
            case 'q':
                std::cerr << "The -q flag is deprecated. The converter is quiet by default." << std::endl;
                break;
            case 'm':
                // only print memory usage and exit
                cmdLineOptions.onlyReportMemory = true;
                break;
            case 'R':
                // only print memory usage and exit
                cmdLineOptions.onlyReportMemoryAndExectime = true;
                break;
            case 'M':
                if (optarg) {
                    cmdLineOptions.memoryLimitInMb = atof(optarg);   
                }
                break;
            case 'T':
                if (optarg) {
                   converter_type = optarg;
                   cmdLineOptions.smartconverter_type = parse_smartconverter_type(converter_type.c_str());
                   printf("DEBUG : smartconverter_type = %d (%s)\n",(int)cmdLineOptions.smartconverter_type,converter_type.c_str());
                }
                break;
            case 'z':
                cmdLineOptions.zMips = true;
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
    
    if (strlen(cmdLineOptions.systemName.c_str()) <= 0 ) {
       getSystemName(cmdLineOptions.systemName);
    }
    
    if (optind >= argc) {
        err = true;
        std::cerr << "Missing input filename parameter." << std::endl;
    } else {
        cmdLineOptions.inputFileName.assign(argv[optind]);
        optind++;
    }
    
    if (argc > optind) {
        err = true;
        std::cerr << "Unexpected additional parameters." << std::endl;
    }
    
    if (SmartConverter::bApproximateCubeHistogram) {
        if (cmdLineOptions.smartconverter_type == eSmartConverterSpatialParallel || cmdLineOptions.smartconverter_type == eSmartConverterChannelParallelTwoPass) {
            err = true;
            std::cerr << "Approximate 3D histogram is not supported in " << converter_type.c_str() << " converter." << std::endl;
        }
    }

    if (cmdLineOptions.maxSwizzledChunkMb > 0) {
        if (!Converter::rotatedDatasetChunking) {
            std::cerr << "WARNING: -L has no effect without -C or -F" << std::endl;
        } else if (!Converter::rotatedChunkOverride.empty()) {
            std::cerr << "WARNING: -L is ignored when explicit chunk dims are given with -K" << std::endl;
        }
    }
            
    if (err) {
        std::cerr << std::endl << usage.str() << std::endl;
        return false;
    }
    
    if (cmdLineOptions.outputFileName.empty()) {
        auto fitsIndex = cmdLineOptions.inputFileName.find_last_of(".fits");
        if (fitsIndex != std::string::npos) {
            cmdLineOptions.outputFileName = cmdLineOptions.inputFileName.substr(0, fitsIndex - 4);
            cmdLineOptions.outputFileName += ".hdf5";
        } else {
            cmdLineOptions.outputFileName = cmdLineOptions.inputFileName + ".hdf5";
        }
    }
    
    return true;
}

void printOptions()
{
   std::cout << "##########################################" << std::endl;
   std::cout << "PARAMETERS:" << std::endl;
   std::cout << "Approximations:" << std::endl;
   std::cout << "\tApproximate histogram: " << SmartConverter::bApproximateCubeHistogram << std::endl;
   std::cout << "\tChunk rotated dataset if it is worth : " << Converter::rotatedDatasetChunking << std::endl;
   std::cout << "\tForce chunking rotated dataset       : " << Converter::rotatedDatasetChunkingForce << std::endl;
   std::cout << "\tMax rotated chunk size               : " << Converter::maxSwizzledChunkBytes / 1e6 << " MB" << std::endl;
   std::cout << "\tExplicit rotated chunk dims (-K)     : ";
   if (Converter::rotatedChunkOverride.empty()) std::cout << "automatic"; else std::cout << Converter::rotatedChunkOverride;
   std::cout << std::endl;
   std::cout << "##########################################" << std::endl;
}

int main(int argc, char** argv) {
    bool progress(false);    
    commandLineOptions cmdLineOptions;    
    
    if (!getOptions(argc, argv, cmdLineOptions)) {
        return 1;
    }
    
    printOptions();

    if (cmdLineOptions.slow && cmdLineOptions.zMips){
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
    if (cmdLineOptions.memoryLimitInMb > 0) {
       memoryLimit = cmdLineOptions.memoryLimitInMb * 1e6; // converting from MB to bytes
    }
    
    std::unique_ptr<Converter> converter;
        
    try {
        converter = Converter::getConverter(cmdLineOptions.inputFileName, cmdLineOptions.outputFileName, 
                                            cmdLineOptions.slow, cmdLineOptions.smart, cmdLineOptions.smartconverter_type, progress, 
                                            cmdLineOptions.zMips, cmdLineOptions.memoryLimitInMb, cmdLineOptions.auto_mode);
        
        if (strlen(cmdLineOptions.systemName.c_str())>0) {
           converter->setSystemName(cmdLineOptions.systemName.c_str());
        }
        
        if (cmdLineOptions.n_io_blocks>1) {
           converter->setIOBlocks(cmdLineOptions.n_io_blocks);
        }
        
        // needs to be before memory and I/O and compute cost report 
        // as this function also selects the most optimal algorithm
        // so the report is based on what is decided here:
        if( !converter->checkMemoryUsage(cmdLineOptions.n_io_blocks, memoryLimit, cmdLineOptions.auto_mode) ) {
            if (!cmdLineOptions.onlyReportMemoryAndExectime && !cmdLineOptions.onlyReportMemory) {
               // only exit in the full execution mode, not in report-only mode
               return 1;
            }
        }

        // always print the memory and exec time report before starting processing
        // just to show the initial estimates for comparison with the actual results
        converter->reportMemoryAndExecTime();
        
        if (cmdLineOptions.onlyReportMemoryAndExectime) {
            return 0;
        } else {
            if (cmdLineOptions.onlyReportMemory) {
                converter->reportMemoryUsage();
                return 0;
            }
        }
                
        DEBUG(std::cout << "Converting FITS file " << cmdLineOptions.inputFileName << " to HDF5 file " << cmdLineOptions.outputFileName << (slow ? " using slower, memory-efficient method" : "") << std::endl;);

        converter->convert();
    } catch (const char* msg) {
        std::cerr << "Error: " << msg << ". Aborting." << std::endl;
        return 1;
    }

    return 0;
}

