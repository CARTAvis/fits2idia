/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"

bool Converter::rotatedDatasetChunking = false;
bool Converter::rotatedDatasetChunkingForce = false;
hsize_t Converter::maxSwizzledChunkBytes = 64000000ULL; // 64 MB default
std::vector<hsize_t> Converter::rotatedChunkOverride;

Converter::Converter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) : timer(), progress(progress), zMips(zMips), swapStokesFreqAxis(false), n_io_blocks(1) {
    TIMER(timer.start("Setup"););
    
    openFitsFile(&inputFilePtr, inputFileName);
    
    long dims[4];
    
    getFitsDims(inputFilePtr, N, dims);
    
    // check relevant FITS keywords if the axis order is STOKES,FREQ -> swap is required:
    checkIfSwapAxisRequired();
        
    stokes = N == 4 ? dims[3] : 1;
    depth = N >= 3 ? dims[2] : 1;
    if ( swapStokesFreqAxis ) {
       stokes = N >= 4 ? dims[2] : 1;
       depth = N == 4 ? dims[3] : 1;
    }
    height = dims[1];
    width = dims[0];
    
    swizzledName = N == 3 ? "ZYX" : "ZYXW";    
    standardDims = trimAxes({stokes, depth, height, width}, N);
    tileDims = trimAxes({1, 1, TILE_SIZE, TILE_SIZE}, N);
    
    numBins = int(std::max(std::sqrt(width * height), 2.0));
    DEBUG(std::cout << "Stokes = " << stokes << ", depth = " << depth << ", image dimensions " << height << " x " << width << " , swizzledName = " << swizzledName.c_str() << " , numBins = " << numBins << std::endl;);
    
    // STATS OBJECTS

    auto statsXYDims = trimAxes({stokes, depth}, N - 2);
    statsXY = Stats(statsXYDims, numBins);
    
    if (depth > 1) {
        swizzledDims = trimAxes({stokes, width, height, depth}, N);
        statsZ = Stats(trimAxes({stokes, height, width}, N - 1));
        auto statsXYZDims = trimAxes({stokes}, N - 3);
        statsXYZ = Stats(statsXYZDims, numBins);
    }
    
    // MIPMAPS
    mipMaps = MipMaps(standardDims, tileDims, zMips);
    
    // Prepare output file
    this->outputFileName = outputFileName;
    tempOutputFileName = outputFileName + ".tmp";        
}

Converter::~Converter() {
    // TODO this is probably unnecessary; the file object destructor should close the file properly.
    outputFile.close();
    closeFitsFile(inputFilePtr);
}

bool Converter::includeDataset(const char* dataset)
{
   if (output_datasets.empty()) {
      return true;
   }

   auto it = output_datasets.find(dataset);   
   if (it != output_datasets.end()) {
      return it->second;
   }
   
   return false;
}

void Converter::ParseExcludeIncludeOptions( std::string exclude_list, std::string include_list ) {
   output_datasets.clear();
   
   std::stringstream ss(include_list);
   std::string item;
   
   // first parse list of includes:
   if( include_list.size() ) {
      // std::getline with ',' extracts tokens directly
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            output_datasets[item] = true;
        }
      }
   }
   
   // now parse the excluded datasets:
   std::stringstream ss2(exclude_list);
   if( exclude_list.size() ) {
      // std::getline with ',' extracts tokens directly
      while (std::getline(ss2, item, ',')) {
        if (!item.empty()) {
            // output_datasets[item] = true;
            auto it = output_datasets.find(item);
            
            // if found in the list set flag to false, otherwise do nothing as the
            // specific dataset will not be present in the list of included datasets:
            if (it != output_datasets.end()) {
              it->second = false;
            }
        }
      }
   }
   
}

std::unique_ptr<Converter> Converter::getOptimalConverter(std::string inputFileName, std::string outputFileName, bool slow, bool smart, eSmartConverterType smarttype, bool progress, bool zMips, int memoryLimitInMb, bool auto_mode) {
    hsize_t memoryLimit  = memoryLimitInMb*1e6; // memory limit in bytes
    std::unique_ptr<SlowConverter> slow_ptr = std::unique_ptr<SlowConverter>(new SlowConverter(inputFileName, outputFileName, progress, zMips));
               
    // first check what is the minimum requirement of the slow converter (one channel at a time no parallelisation) :
    MemoryUsage slow_converter_memory = slow_ptr->calculateMemoryUsage();
    double slow_converter_memory_mb = slow_converter_memory.total/1e6;
                
    // safety buffer factor - how much more memory is required than calculated, currently 10%
    double safety_buffer_factor = 1.1; 
    
    // maximum numbe of iterations when checking memory limits:
    int max_iter = 10;
               
    double single_channel_image_mb = slow_ptr->width*slow_ptr->height*sizeof(float)/1e6;
    double cube_size_mb = slow_ptr->depth*slow_ptr->width*slow_ptr->height*sizeof(float)/1e6;
               
    if ( memoryLimitInMb > (slow_converter_memory_mb*1.1) ) { 
       // use this path if we have at least enough memory for SlowConverter
       std::unique_ptr<SmartFastConverter> smartfast_ptr = std::unique_ptr<SmartFastConverter>(new SmartFastConverter(inputFileName, outputFileName, progress, zMips)); 
       bool can_use_smartfast = smartfast_ptr->ReduceMemoryUsage( memoryLimit, max_iter );
       MemoryUsage smartfast_converter_memory = smartfast_ptr->calculateMemoryUsage();
       double smartfast_converter_memory_mb = smartfast_converter_memory.total/1e6;
                  
       SmartConverter* smart_ptr = dynamic_cast<SmartConverter*>(smartfast_ptr.get());
       bool can_use_smart = smart_ptr->ReduceMemoryUsage( memoryLimit, max_iter ); 
       MemoryUsage smart_converter_memory = smartfast_ptr->calculateMemoryUsage();
       double smart_converter_memory_mb = smartfast_converter_memory.total/1e6;
       
       if (!can_use_smartfast && !can_use_smart ) { // was empirical : if ( cube_size_mb >= 5000 && memoryLimitInMb < 10000 ) { BASED ON TESTS : if image is at least 5GB and available memory <= 12 GB:
          // if cannot use any of the smart converters -> use slow converter:
          std::cout << "AUTOMATIC ALGORITHM SELECTION : using SlowConverter as there is not enough memory to use Smart or SmartFast converter" << std::endl;
          return slow_ptr;
       } else {
         // we can use at least one of the smart converters :
         
         std::cout << "AUTOMATIC ALGORITHM SELECTION : returning Smart or " << std::endl;
         if (can_use_smartfast && can_use_smart ) {
            // if both can be used, calculate expected cost/execution time and decide based on this :
            // TODO :
            // CompCost smartfast_comp_cost = smartfast_ptr->calculateCompCost();
            // CompCost smart_comp_cost = smart_ptr->calculateCompCost();
            // std::cout << "AUTOMATIC ALGORITHM SELECTION : using SmartFastConverter" << std::endl;
            // if ( smartfast_comp_cost < smart_comp_cost ) { return smartfast_ptr } else { return smart_ptr }
            std::cout << "ERROR : AUTOMATIC ALGORITHM SELECTION : decision based on computational cost not yet implemented -> using SmartFastConverter" << std::endl;
            return smartfast_ptr;
         } else {
            if (can_use_smartfast) {
               std::cout << "AUTOMATIC ALGORITHM SELECTION : using SmartFastConverter" << std::endl;
               return smartfast_ptr;
            } else {
               std::cout << "AUTOMATIC ALGORITHM SELECTION : using SmartConverter" << std::endl;
               return std::unique_ptr<Converter>(smart_ptr);
               // return std::unique_ptr<Converter>(new SmartConverter(inputFileName, outputFileName, progress, zMips));
            }
         }
       }
    } else {
       // if we do not even have enough memory for a the SlowConverter (total minimum - one image at a time), use 
       // partial converter reading 1 row at a time:
       std::cout << "AUTOMATIC ALGORITHM SELECTION : using MicroMemoryConverter" << std::endl;
       return std::unique_ptr<Converter>(new MicroMemoryConverter(inputFileName, outputFileName, progress, zMips));
    }

    // nothing optimal identified use SlowConverter as a fallback:
    return slow_ptr;
}

std::unique_ptr<Converter> Converter::getConverter(std::string inputFileName, std::string outputFileName, bool slow, bool smart, eSmartConverterType smarttype, bool progress, bool zMips, int memoryLimitInMb, bool auto_mode) {
    if (slow) {
        std::cout << "DEBUG : using SlowConverter object" << std::endl;
        return std::unique_ptr<Converter>(new SlowConverter(inputFileName, outputFileName, progress, zMips));
     } else if (smart) { // was also && memoryLimitInMb > 0
        std::cout << "DEBUG : using SmartConverter object with memory limit = " << memoryLimitInMb << " MB." << " smarttype = " << smarttype << std::endl;
        SmartConverter* ptr = NULL;
        if (smarttype == eAutoSelectedSmartConverter) {
           std::cout << "DEBUG : using AutoSelectedSmartConverter object" << std::endl;
           if( auto_mode ){
               return getOptimalConverter(inputFileName, outputFileName, slow, smart, smarttype, progress, zMips, memoryLimitInMb, auto_mode);
           } else {
              // otherwise use the default one:
              ptr = new SmartConverter(inputFileName, outputFileName, progress, zMips);
           }
        } else {
            if (smarttype == eSmartConverterChannelParallelTwoPass) {
                ptr = new SmartFastTwoPassConverter(inputFileName, outputFileName, progress, zMips);
                std::cout << "DEBUG : using SmartFastTwoPassConverter object" << std::endl;
            } else if (smarttype == eSmartConverterChannelParallel) {
                ptr = new SmartFastConverter(inputFileName, outputFileName, progress, zMips);
                std::cout << "DEBUG : using SmartFastConverter object" << std::endl;
            } else if (smarttype == eSmartMicroMemoryConverter) {
                ptr = new MicroMemoryConverter(inputFileName, outputFileName, progress, zMips);
                std::cout << "DEBUG : using MicroMemoryConverter object" << std::endl;
            } else {
                ptr = new SmartConverter(inputFileName, outputFileName, progress, zMips);
                std::cout << "DEBUG : using SmartConverter object" << std::endl;
            }
        }
        ptr->setMemoryLimit(memoryLimitInMb);
        std::unique_ptr<Converter> pSmartConverter(ptr);
        return pSmartConverter;
    } else {
        if (memoryLimitInMb > 0) {
            // Fast path with a memory limit: rotate in strips so the whole rotated cube
            // (and full-size Z stats) never have to be in memory. Identical to FastConverter
            // when the limit is large enough (single strip).
            std::cout << "DEBUG : using FastConverterLimitedMemory object with memory limit = " << memoryLimitInMb << " MB" << std::endl;
            FastConverterLimitedMemory* ptr = new FastConverterLimitedMemory(inputFileName, outputFileName, progress, zMips);
            ptr->setMemoryLimit(memoryLimitInMb);
            return std::unique_ptr<Converter>(ptr);
        }
        std::cout << "DEBUG : using FastConverter object" << std::endl;
        return std::unique_ptr<Converter>(new FastConverter(inputFileName, outputFileName, progress, zMips));
    }
}

void Converter::setSystemName(const char* system_name )
{
   if (system_name && system_name[0]) {
      systemName = system_name;
   }
}

void Converter::reportMemoryAndExecTime()
{
   reportMemoryUsage();
   
   reportExecTime();
}

void Converter::reportExecTime()
{
   // predict execution time:
   IOCostModel readModel, writeModel;
   get_io_cost_model(systemName.c_str(), readModel, writeModel ); // NULL -> SystemName, for example "SETONIX" to get specific BW
   IOCostBreakdown iocost = estimateIO(stokes, depth, height, width, numBins, readModel, writeModel );
   iocost.print();
}

void Converter::reportMemoryUsage() {
    MemoryUsage m = calculateMemoryUsage();
    
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;    
    hsize_t maxAllocationBytes = 0;
    std::string maxAllocationDataset = "";
    for (auto& kv : m.sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
        if (kv.second > maxAllocationBytes) {
            maxAllocationDataset = kv.first;
            maxAllocationBytes = kv.second;
        }
    }

    std::cout << "MAX ALLOCATION:\t" << maxAllocationBytes * 1e-9 << "GB" << " required for " << maxAllocationDataset << std::endl;    
    std::cout << "TOTAL ALLOCATION  :\t" << m.total * 1e-9 << "GB" << m.note << std::endl;
    double total_ram_gb = m.total * 1e-9 * 1.1;
    std::cout << "TOTAL RAM REQUIRED:\t" << "Add additional 10% of RAM or SLURM job limit: --mem " << total_ram_gb << "GB" << std::endl;    
}

void Converter::getDimensions( hsize_t& _stokes, hsize_t& _depth, hsize_t& _height, hsize_t& _width ) {
   _stokes = stokes;
   _depth  = depth;
   _height = height;
   _width  = width;
}

bool Converter::ReduceMemoryUsage( hsize_t memoryLimit, int max_iter /*=10*/ ) {
   std::cout << "ERROR : function is not implemented for this class" << std::endl;
   return false;
}

bool Converter::checkIfSwapAxisRequired() {
    int numAttributes;
    readFitsHeader(inputFilePtr, numAttributes);

    bool bFreqAxisFound = false;

    swapStokesFreqAxis = false;
    // Check if both STOKES and FREQ axis are present and the order is STOKES,FREQ:
    for (int i = 1; i <= numAttributes; i++) {
       std::string attributeName;
       std::string attributeValue;
       readFitsAttribute(inputFilePtr, i, attributeName, attributeValue);
       DEBUG(std::cout << "|" << attributeName.c_str() << "| = |" << attributeValue.c_str() << "|" << std::endl;);

       if (attributeName == "CTYPE3" && attributeValue.find("STOKES") != std::string::npos) {
          std::cout << "INFO : detected 3rd axis CTYPE3 = STOKES" << std::endl;
          swapStokesFreqAxis = true;
       }
       if (attributeName == "CTYPE4" && attributeValue.find("FREQ") != std::string::npos) {
          std::cout << "INFO : detected 4th axis CTYPE4 = FREQ" << std::endl;
          bFreqAxisFound = true;
       }
    }
    if (swapStokesFreqAxis) {
       if (!bFreqAxisFound) {
          swapStokesFreqAxis = false; // If Stokes is 3rd axis, but 4th axis is not frequency -> no swap
       }
    }
    std::cout << "INFO : swapStokesFreqAxis = " << swapStokesFreqAxis << std::endl;
    
    return swapStokesFreqAxis;
}

void Converter::DebugDimsAndParameters( const std::vector<hsize_t>& swizzledDims, const std::vector<hsize_t>& swizzledChunkDims, const H5::DataSet& swizzledDataSet ) {
   // DEBUG: verify swizzledDims and swizzledChunkDims line up axis-for-axis
   std::cout << "DEBUG swizzledDims       = [";
   for (auto d : swizzledDims) std::cout << d << " ";
   std::cout << "]" << std::endl;
        
   std::cout << "DEBUG swizzledChunkDims  = [";
   for (auto d : swizzledChunkDims) std::cout << d << " ";
   std::cout << "]" << std::endl;
        
   H5::DSetCreatPropList actualPropList = swizzledDataSet.getCreatePlist();
   if (actualPropList.getLayout() == H5D_CHUNKED) {
      int rank = swizzledDims.size();
      std::vector<hsize_t> actualChunkDims(rank);
      actualPropList.getChunk(rank, actualChunkDims.data());

      std::cout << "DEBUG actual chunk dims from dataset = [";
      for (auto d : actualChunkDims) std::cout << d << " ";
          std::cout << "]" << std::endl;
   } else {
      std::cout << "DEBUG WARNING: swizzledDataSet is NOT chunked!" << std::endl;
   }
       
   // --- DEBUG: print current chunk cache settings for swizzledDataSet ---
   H5::DSetAccPropList currentAccessPlist = swizzledDataSet.getAccessPlist();
   size_t rdccNumSlots = 0;
   size_t rdccNumBytes = 0;
   double rdccW0 = 0.0;
   currentAccessPlist.getChunkCache(rdccNumSlots, rdccNumBytes, rdccW0);

   std::cout << "DEBUG chunk cache: nslots=" << rdccNumSlots
             << " nbytes=" << rdccNumBytes << " (" << (rdccNumBytes / 1e6) << " MB)"
             << " w0=" << rdccW0 << std::endl;
   // --- END DEBUG ---

}

const std::vector<hsize_t>& Converter::getSwizzledChunkDims() {
    if (!swizzledChunkDimsChosen) {
        swizzledChunkDims4 = chooseSwizzledChunkDims();
        swizzledChunkDimsChosen = true;
    }
    return swizzledChunkDims4;
}

std::vector<hsize_t> Converter::chooseSwizzledChunkDims() {
    if (!rotatedDatasetChunking || depth <= 1) return {};   // -F and -K set rotatedDatasetChunking too

    // FAST / FAST-LIMITED-MEMORY write whole cubes or full-height strips: already contiguous, never chunk
    if (strncmp(getConverterType(), "FAST", 4) == 0) {
        std::cout << "INFO: rotated dataset chunking (-C/-F/-K) is not used by the " << getConverterType()
                  << " converter (its writes are already contiguous)" << std::endl;
        return {};
    }

    // -K: explicit chunk shape -- no read guard, no sizing; write comparison printed for information only
    if (!rotatedChunkOverride.empty()) {
        const hsize_t cw = std::min(rotatedChunkOverride[0], width);
        const hsize_t ch = std::min(rotatedChunkOverride[1], height);
        const hsize_t cd = (rotatedChunkOverride.size() > 2) ? std::min(rotatedChunkOverride[2], depth) : depth;
        std::vector<hsize_t> chunkDims4 = {1, cw, ch, cd};
        const hsize_t chunkBytes = cw * ch * cd * sizeof(float);

        if (chunkBytes >= (4ULL << 30)) {
            throw "Rotated chunk requested with -K is >= 4 GB (HDF5 chunk size limit)";
        }
        if (cw != rotatedChunkOverride[0] || ch != rotatedChunkOverride[1] ||
            (rotatedChunkOverride.size() > 2 && cd != rotatedChunkOverride[2])) {
            std::cout << "INFO: -K chunk dims clipped to the dataset dimensions" << std::endl;
        }
        
        // NEW: chunks should be written whole by one rotation tile (TILE_SIZE x TILE_SIZE x depth).
        // OK if the chunk dim divides TILE_SIZE, or if the image is no larger than one tile in that
        // direction and the chunk spans it entirely.
        auto alignedToTiles = [](hsize_t c, hsize_t dim) {
            return (TILE_SIZE % c == 0) || (dim <= TILE_SIZE && c == dim);
        };
        if (!alignedToTiles(cw, width) || !alignedToTiles(ch, height)) {
            std::cout << "WARNING: -K chunk " << cw << "x" << ch << " is not aligned with the " << TILE_SIZE << "x" << TILE_SIZE
                      << " rotation tiles -> chunks will be written partially by several tiles (slower writes)" << std::endl;
        }
        
        const hsize_t chunksPerSpectrum = (depth + cd - 1) / cd;
        std::cout << "INFO: -K: explicit rotated chunk dims " << chunkDims4 << " = " << chunkBytes / 1e6 << " MB, "
                  << chunksPerSpectrum << " chunk(s) per spectrum" << std::endl;
        if (chunksPerSpectrum > 4) {
            std::cout << "WARNING: -K: each spectrum spans " << chunksPerSpectrum << " chunks, expect slow spectral profiles" << std::endl;
        }

        IOCostModel seqRead, seqWrite, randRead, randWrite;
        get_io_cost_model(systemName.c_str(), seqRead, seqWrite, false);
        get_io_cost_model(systemName.c_str(), randRead, randWrite, true);
        chunkingPaysOffOnWrite({stokes, width, height, depth}, chunkDims4, stokes, depth, height, width,
                               seqWrite, randWrite); // result ignored -- printed for comparison only
        return chunkDims4;
    }
        
    const bool force = rotatedDatasetChunkingForce;

    // only SMART-CHAN-PARALLEL writes partial-depth slabs; SLOW, SMART-XY, 2PASS write full-depth tiles
    const bool partialDepthWriter = (strcmp(getConverterType(), "SMART-CHAN-PARALLEL") == 0);
    const hsize_t chunkDepth = partialDepthWriter ? std::max((hsize_t)1, (hsize_t)n_io_blocks) : depth;

    // 1. read-side guard: a cursor spectrum must not span more than a few chunks (-F overrides with a warning)
    const hsize_t MAX_CHUNKS_PER_SPECTRUM = 4;
    const hsize_t chunksPerSpectrum = (depth + chunkDepth - 1) / chunkDepth;
    if (chunksPerSpectrum > MAX_CHUNKS_PER_SPECTRUM) {
        if (!force) {
            std::cout << "INFO: rotated dataset kept contiguous: chunk depth " << chunkDepth
                      << " would split each spectrum into " << chunksPerSpectrum << " chunks" << std::endl;
            return {};
        }
        std::cout << "WARNING: -F: forcing chunk depth " << chunkDepth << " -> each spectrum spans "
                  << chunksPerSpectrum << " chunks, expect slow spectral profiles" << std::endl;
    }

    IOCostModel seqRead, seqWrite, randRead, randWrite;
    get_io_cost_model(systemName.c_str(), seqRead, seqWrite, false);
    get_io_cost_model(systemName.c_str(), randRead, randWrite, true);

    // 2. chunk sizing: cap set by the read side, shrink x first (CARTA reads favour y-generous shapes)
    hsize_t cw = std::min((hsize_t)TILE_SIZE, width), ch = std::min((hsize_t)TILE_SIZE, height);
    while (cw * ch * chunkDepth * sizeof(float) > maxSwizzledChunkBytes && (cw > 1 || ch > 1)) {
        if (cw > 16 || ch == 1) cw = std::max((hsize_t)1, cw / 2);
        else                     ch = std::max((hsize_t)1, ch / 2);
    }
    std::vector<hsize_t> chunkDims4 = {1, cw, ch, chunkDepth};

    if (cw * ch * chunkDepth * sizeof(float) > maxSwizzledChunkBytes) {
        std::cout << "WARNING: chunk size limit " << maxSwizzledChunkBytes / 1e6 << " MB is smaller than one "
                  << chunkDepth << "-channel spectrum -> using " << chunkDims4 << std::endl;
    }


    const hsize_t chunkBytes = cw * ch * chunkDepth * sizeof(float);
    std::cout << "INFO: candidate rotated chunk " << chunkDims4 << " = " << chunkBytes / 1e6 << " MB"
              << " (cap " << maxSwizzledChunkBytes / 1e6 << " MB, write plateau "
              << plateauBytes(seqWrite) / 1e6 << " MB)" << std::endl;

    // 3. write-side gate (full-depth tile writers only). Always printed; decides only without -F.
    if (!partialDepthWriter) {
        const std::vector<hsize_t> swizzledDims4 = {stokes, width, height, depth};
        bool payOff = chunkingPaysOffOnWrite(swizzledDims4, chunkDims4, stokes, depth, height, width, seqWrite, randWrite);
        if (!payOff) {
            if (!force) return {};
            std::cout << "INFO: -F: chunking although the predicted write speedup is below the threshold" << std::endl;
        }
    }

    std::cout << "INFO: chunking rotated dataset with chunk dims " << chunkDims4
              << (force ? " (forced, -F)" : "") << std::endl;
    return chunkDims4;
}

void Converter::convert() {
    // CREATE OUTPUT FILE
    
    // TODO dataset variables should be local and passed into the copy function?
    auto start = std::chrono::high_resolution_clock::now();
    
    outputFile = H5::H5File(tempOutputFileName, H5F_ACC_TRUNC);
    outputGroup = outputFile.createGroup("0");
    
    std::vector<hsize_t> chunkDims;
    if (useChunks(standardDims)) {
        chunkDims = tileDims;
    }
    
    H5::FloatType floatType(H5::PredType::NATIVE_FLOAT);
    floatType.setOrder(H5T_ORDER_LE);
    createHdf5Dataset(standardDataSet, outputGroup, "DATA", floatType, standardDims, chunkDims);
    
    statsXY.createDatasets(outputGroup, "XY");

    if (depth > 1) {
        statsXYZ.createDatasets(outputGroup, "XYZ");
        // statsZ.createDatasets(outputGroup, "Z");
        hsize_t zChunkHeight = std::min((hsize_t)TILE_SIZE, height);
        hsize_t zChunkWidth  = std::min((hsize_t)TILE_SIZE, width);
        statsZ.createDatasets(outputGroup, "Z", trimAxes({1, zChunkHeight, zChunkWidth}, N - 1));
        
        auto swizzledGroup = outputGroup.createGroup("SwizzledData");
        // We use this name in papers because it sounds more serious. :)
        outputGroup.link(H5L_TYPE_HARD, "SwizzledData", "PermutedData");
        
        const auto& chunk4 = getSwizzledChunkDims();
        auto swizzledChunkDims = chunk4.empty() ? EMPTY_DIMS : trimAxes(chunk4, N);
        createHdf5Dataset(swizzledDataSet, swizzledGroup, swizzledName, floatType, swizzledDims, swizzledChunkDims);
        DebugDimsAndParameters(swizzledDims, swizzledChunkDims, swizzledDataSet);
    }
    
    mipMaps.createDatasets(outputGroup);
    
    // COPY HEADERS
    
    TIMER(timer.start("Headers"););
    
    writeHdf5Attribute(outputGroup, "SCHEMA_VERSION", std::string(SCHEMA_VERSION));
    writeHdf5Attribute(outputGroup, "HDF5_CONVERTER", std::string(HDF5_CONVERTER));
    writeHdf5Attribute(outputGroup, "HDF5_CONVERTER_VERSION", std::string(HDF5_CONVERTER_VERSION));
    writeHdf5Attribute(outputGroup, "HDF5_CONVERTER_TYPE", getConverterType() );
    writeHdf5Attribute(outputGroup, "HDF5_PARAM_CUBE_HISTOGRAM_APPROXIMATE", getCubeHistogramApproximated() );

    int numAttributes;
    readFitsHeader(inputFilePtr, numAttributes);
    
    std::vector<std::string> wcsStokesKeywords = {"CTYPE3","CRVAL3","CDELT3","CRPIX3","CUNIT3"};
    std::vector<std::string> wcsFreqKeywords = {"CTYPE4","CRVAL4","CDELT4","CRPIX4","CUNIT4"};
    
    // IMPORTANT: This is 1-indexed!
    for (int i = 1; i <= numAttributes; i++) {
        bool bSwappingStokesFreqAxisNow = false;
        std::string attributeName;
        std::string attributeValue;
        readFitsAttribute(inputFilePtr, i, attributeName, attributeValue);
 
        if (swapStokesFreqAxis) { // check if swap of STOKES <-> FREQ axis is required:
           // If Stokes is 3rd axis and Freq 4th axis -> SWAP Stokes and Frequency axis in FITS header
           if (std::find(wcsStokesKeywords.begin(), wcsStokesKeywords.end(), attributeName) != wcsStokesKeywords.end()) {
              if(attributeName.back() == '3') {
                 // if swap of STOKES and FREQ axis is required change 3 -> 4:
                 std::cout << "INFO : swapping attribute " << attributeName << " (3 -> 4) " << std::endl;
                 attributeName.back() = '4';
                 bSwappingStokesFreqAxisNow = true;
              }
           }else{
              if ( std::find(wcsFreqKeywords.begin(), wcsFreqKeywords.end(), attributeName) != wcsFreqKeywords.end() ){
                 if(attributeName.back() == '4') {
                    // if swap of STOKES and FREQ axis is required change 4 -> 3:
                    std::cout << "INFO : swapping attribute " << attributeName << " (4 -> 3) " << std::endl;
                    attributeName.back() = '3';
                    bSwappingStokesFreqAxisNow = true;
                 }
              }
           }
        }
        
        if (attributeName.empty() || attributeName.find("COMMENT") == 0 || attributeName.find("HISTORY") == 0) {
            // TODO we should actually do something about these
        } else {
            if (outputGroup.attrExists(attributeName)) {
                std::cout << "Warning: Skipping duplicate attribute '" << attributeName << "'" << std::endl;
            } else {
                bool parsingFailure(false);
                
                if (attributeValue.length() >= 2 && attributeValue.find('\'') == 0 &&
                    attributeValue.find_last_of('\'') == attributeValue.length() - 1) {
                    // STRING
                    std::string attributeValueStr;
                    readFitsStringAttribute(inputFilePtr, attributeName, attributeValueStr);
                    // this may not be required 
                    if( bSwappingStokesFreqAxisNow ) {
                       // if swapping axis now, we have to use attributeValue as the attributes's name has been swapped 
                       // for example CTYPE3 -> CTYPE4 which means value of CTYPE4 would be read (FREQ) and no swap would happen!
                       attributeValueStr = attributeValue;
                    }
                    writeHdf5Attribute(outputGroup, attributeName, attributeValueStr);
                    DEBUG(std::cout << "Written attribute " << attributeName.c_str() << " = " << attributeValueStr.c_str() << std::endl;);
                } else if (attributeValue == "T" || attributeValue == "F") {
                    // BOOLEAN
                    bool attributeValueBool = (attributeValue == "T");
                    writeHdf5Attribute(outputGroup, attributeName, attributeValueBool);
                } else if (attributeValue.find('.') != std::string::npos) {
                    // TRY TO PARSE AS DOUBLE
                    try {
                        double attributeValueDouble = std::stod(attributeValue);
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueDouble);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as a float." << std::endl;
                        parsingFailure = true;
                    } catch (const std::out_of_range& e) {
                        // Special handling for subnormal numbers
                        long double attributeValueLongDouble = std::stold(attributeValue);
                        double attributeValueDouble = (double) attributeValueLongDouble;
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueDouble);
                        
                        std::ostringstream ostream;
                        ostream.precision(13);
                        ostream << attributeValueDouble;
                        std::string original(attributeValue);
                        std::string round_trip(ostream.str());
                        transform(original.begin(), original.end(), original.begin(), ::toupper);
                        transform(round_trip.begin(), round_trip.end(), round_trip.begin(), ::toupper);
                                                
                        if (original != round_trip) {
                            std::cout << "Warning: the value of attribute  '" << attributeName << "' is not representable as a normalised double precision floating point number. Some precision has been lost.\nOriginal string representation:\n'" << original << "'\nFinal string representation:\n'" << round_trip << "'" << std::endl;
                        }
                    }
                } else {
                    // TRY TO PARSE AS INTEGER
                    try {
                        int64_t attributeValueInt = std::stoi(attributeValue);
                        writeHdf5Attribute(outputGroup, attributeName, attributeValueInt);
                    } catch (const std::invalid_argument& ia) {
                        std::cout << "Warning: could not parse attribute '" << attributeName << "' as an integer." << std::endl;
                        parsingFailure = true;
                    }
                }
                
                if (parsingFailure) {
                    // FALL BACK TO STRING
                    writeHdf5Attribute(outputGroup, attributeName, attributeValue);
                }
            }
        }
    }
    
    // MAIN CONVERSION AND CALCULATION FUNCTION

    copyAndCalculate();
            
    TIMER(timer.print(product(standardDims)););
    
    // Rename from temp file
    rename(tempOutputFileName.c_str(), outputFileName.c_str());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = ms_d(end - start);
    std::cout << "Execution of entire Converter::convert took: " << duration.count() << " milliseconds = " << duration.count()/1000.00 << " seconds" << std::endl;
}


IOCostBreakdown Converter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel)
{
   std::cout << "ERROR : virtual method Converter::estimateIO not implemented in this class !" << std::endl;
   
   IOCostBreakdown result;
   return result;
}

