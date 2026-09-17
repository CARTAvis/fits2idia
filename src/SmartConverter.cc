/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"
#include <algorithm> // for std::min

// 1. Conditionally include the OpenMP header at the top of your file
#ifdef _OPENMP
    #include <omp.h>
#endif

// flag to enable single pass through the data 
// XYZ (cube) histogram is calculated using channel histograms 
// which means it's approximate only, but this is "good enough" for the visualisation purposes
bool SmartConverter::bApproximateCubeHistogram = false;

SmartConverter::SmartConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) 
 : Converter(inputFileName, outputFileName, progress, zMips), allowed_mipmaps_threads(1), min_mipmap_threads(8), height_divider(1)
{
   SetChunkDivider(height_divider);
#ifdef _OPENMP
   allowed_mipmaps_threads = omp_get_max_threads();
#endif       
   
}

MemoryUsage SmartConverter::calculateMemoryUsage() {
    MemoryUsage m;
    
    std::cout << "MEMORY_ESTIMATE: (SmartConverter::calculateMemoryUsage) parameters: n_io_blocks = " << n_io_blocks << " , height = " << height << " , width = " << width << std::endl;

    // memory used in pass 1 :
    m.sizes["Main dataset"] = n_io_blocks * height * width * sizeof(float); // multiple (_sliceIncrement) image slices can be read in 1 block , TICKED 
    std::cout << "MEMORY_ESTIMATE: Main dataset  = " << m.sizes["Main dataset"]* 1e-9 << " GB" << std::endl;
    
    m.sizes["XY stats"] = Stats::size({depth}, numBins, height_chunk); // height added - has to agree with statsXY.createBuffers({depth}, height_chunk);, TICKED
    std::cout << "MEMORY_ESTIMATE: XY stats  = " << m.sizes["XY stats"]* 1e-9 << " GB" << std::endl; 
    
    // MipMaps in multiple threads:
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {1, height, width}, zMips);
    if( allowed_mipmaps_threads > 1 ) {
       // times number of threads (see function SmartConverter::calculateChannelStats) :
       m.sizes["Mipmaps"] *= (1+allowed_mipmaps_threads); // for mipMaps and thread_mipmaps_array
    }
    std::cout << "MEMORY_ESTIMATE: MipMaps calculations (#threads = " << allowed_mipmaps_threads << ") = " << m.sizes["Mipmaps"] * 1e-9 << " GB " << std::endl;
    

    if (depth > 1) {
       m.sizes["XYZ stats"] = Stats::size({}, numBins, height ); // was depth); has to agree with statsXYZ.createBuffers({}, height);
       std::cout << "MEMORY_ESTIMATE: XYZ stats  = " << m.sizes["XYZ stats"]* 1e-9 << " GB" << std::endl;
    }

    hsize_t total_pass1 = 0;
    for (auto& kv : m.sizes) {
        total_pass1 += kv.second;
    }
       
    std::cout << "MEMORY_ESTIMATE: 1st pass = " << total_pass1 * 1e-9 << " GB " << std::endl;
    std::cout << "MEMORY_ESTIMATE: height_chunk = " << height_chunk << std::endl;
    //----------------------------------------------- end of 1st pass -----------------------------------------------

    // 2nd pass (cannot see any specific allocations)
    hsize_t total_pass2 = 0;
    if ( bApproximateCubeHistogram ) {
       std::cout << "MEMORY_ESTIMATE: 2nd pass (approx XYZ histogram) = " << total_pass2 * 1e-9 << " GB " << std::endl;
    }else {
       // 2nd pass does not use any extra memory:
       std::cout << "MEMORY_ESTIMATE: 2nd pass (accurate XYZ histogram) = " << total_pass2 * 1e-9 << " GB " << std::endl;
    }
    
    // rotation pass : standardSlice , rotatedSlice : 
    hsize_t total_rotation_pass = 0;
    if (depth > 1) {
        hsize_t sliceSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
        m.sizes["Rotation"] = 2*sliceSize*sizeof(float);
        std::cout << "MEMORY_ESTIMATE: rotation " << m.sizes["Rotation"] * 1e-9 << " GB " << std::endl;
        
        m.sizes["Z stats"] = Stats::size({TILE_SIZE, TILE_SIZE}); // agrees with statsZ.createBuffers({TILE_SIZE, TILE_SIZE});        
        std::cout << "MEMORY_ESTIMATE: Z stats " << m.sizes["Z stats"] * 1e-9 << " GB " << std::endl;

        total_rotation_pass = m.sizes["Rotation"] + m.sizes["Z stats"];
        std::cout << "MEMORY_ESTIMATE: Rotation = "  << total_rotation_pass * 1e-9 << " GB " << std::endl;
    }


    m.total = std::max( std::max(total_pass1, total_pass2), total_rotation_pass);
    std::cout << "MEMORY_ESTIMATE peak usage = " << m.total * 1e-9 << " GB " << std::endl;
    
    size_t sum_total = total_pass1 + total_pass2 + total_rotation_pass;
    std::cout << "MEMORY_ESTIMATE sum usage = " << m.total * 1e-9 << " GB " << std::endl;
    
    return m;
}

bool SmartConverter::ReduceMemoryUsage( hsize_t memoryLimit, int max_iter /*=10*/ ) {
   hsize_t predictedTotal = calculateMemoryUsage().total;

   std::vector<int> heigth_dividers;
   getDividers(height, heigth_dividers);

   
   // outer most is change of number of threads which we want to avoid as much as possible:
   while( allowed_mipmaps_threads >= 1 ) {          
       int iter = 0;

       // number of iterations is limited by the number of dividers in the array heigth_dividers:
       while (iter < heigth_dividers.size() && iter < max_iter && predictedTotal>memoryLimit ) {
          int divider = heigth_dividers[iter];
          std::cout << "MEMORY REDUCTION : testing divider = " << divider << ", #MipMap_Threads = " << allowed_mipmaps_threads << std::endl;
          SetChunkDivider( divider );
      
          MemoryUsage memusage = calculateMemoryUsage();
          predictedTotal = memusage.total;

          if (predictedTotal <= memoryLimit ) {
              std::cout << "MEMORY MINIMSATION : required predicted memory " << predictedTotal * 1e-9 << "GB below memory limit of " << memoryLimit * 1e-9 << "GB -> exiting loop" << std::endl;
              return true;
          } else {
              std::cout << "MEMORY MINIMSATION : required predicted memory " << predictedTotal * 1e-9 << "GB still exceeds the limit of " << memoryLimit * 1e-9 << "GB (divider = " << divider << ")" << std::endl;
          }
          
          // counts all iterations:
          iter++;
       }
       
       if( allowed_mipmaps_threads > 1 ) {
           // reduce number of threads used for MipMaps as well:
           allowed_mipmaps_threads = int(allowed_mipmaps_threads/2);
           std::cout << "MEMORY REDUCTION : reduced number of allowed_mipmaps_threads to " << allowed_mipmaps_threads << std::endl;
       }else{
           std::cout << "ERROR : reached the minimum possible memory for specified number of n_io_blocks = " << n_io_blocks << std::endl;
           break;
       }
   }
   
   return false;
}

void SmartConverter::SetChunkDivider( int divider ) {
   height_divider = divider;
   height_chunk = height / divider;

   std::cout << "Divider set to " << divider << " and height_chunk = " << height_chunk << std::endl;
}


void SmartConverter::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    
    if(memoryLimitInMb <= 0) {
       std::cerr << "ERROR : memory limit not set for the SmartConverter and it is strictly required -> exiting SmartConverter::copyAndCalculate function" << std::endl;
       return;
    }
    
    // calculate memory limits in different units, slice here is image in a single freq. channel:
    int sliceIncrement = n_io_blocks;
    int sliceIncrementCount = depth/sliceIncrement;
    int leftOverSlices = (depth % sliceIncrement);   
    int n_blocks = sliceIncrementCount;
    if (leftOverSlices > 0) {
       n_blocks++;
    }
    std::cout << "DEBUG : n_blocks = " << n_blocks << " vs.  sliceIncrementCount = " << sliceIncrementCount << " and leftOverSlices = " << leftOverSlices << std::endl;
              
              
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    auto start1 = std::chrono::high_resolution_clock::now();    
    // Allocate one channel at a time, and no swizzled data
    TIMER(timer.start("Allocate"););
    std::cout << "MEMORY : stokes:" << stokes << " depth: " << depth << " TILE_SIZE:" << TILE_SIZE << " N:" << N << std::endl;
    std::cout << "MEMORY (SmartConverter::copyAndCalculate): allocating standardCube with size " << double(height * width * sliceIncrement*sizeof(float))/1e9 << " GB " << std::endl << std::flush;
    standardCube = new float[height * width * sliceIncrement];    
    
    // Allocate one stokes of stats at a time
    // statsXY.createBuffers({depth});
    printf("DEBUG : before statsXY.createBuffers({%llu}, %llu)\n",depth,height);
    statsXY.createBuffers({depth}, height_chunk);
        
    if (depth > 1) {
        printf("DEBUG : before statsXYZ.createBuffers({}, %llu)\n",height);
        statsXYZ.createBuffers({}, height);
    }

    printf("DEBUG : before mipMaps.createBuffers({%d,%llu,%llu})\n",1,depth,height);    
    mipMaps.createBuffers({1, height, width});
    
    // Allocate thread-local MipMaps array once
    thread_mipmaps_array.clear();
    thread_mipmaps_array.reserve(allowed_mipmaps_threads);
    for (int i = 0; i < allowed_mipmaps_threads; ++i) {
        thread_mipmaps_array.emplace_back(standardDims, tileDims, zMips);
        thread_mipmaps_array.back().createBuffers({1, height, width}); 
    }

    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    total_pureprocessing_ms += double(duration1.count());


    hsize_t image_size = width*height;
    
    std::vector<double> savedCubeMin(stokes), savedCubeMax(stokes);

    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        StatsCounter counterXYZ;
        
        double total_first_pass_processing_ms = 0.00;
        for (hsize_t block = 0; block < n_blocks; block++ ) {
            hsize_t c_start = block*sliceIncrement;
            hsize_t c_end   = c_start + sliceIncrement;
            if (block == (n_blocks-1) && leftOverSlices > 0 ) {
               c_end   = c_start + leftOverSlices; // if last block and there are left over slices                
            }
            hsize_t n_channels = (c_end-c_start);
            hsize_t block_size = n_channels*height*width;
            std::cout << "DEBUG : processing block = " << block << " -> channel range " << c_start << " - " << c_end << std::endl;
            
            std::cout << "DEBUG : reading from c_start = " << c_start << " blockSize = " <<  block_size << " bytes" << std::endl;
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            // measuring I/O :
            auto start_io = std::chrono::high_resolution_clock::now();
            readFitsData(inputFilePtr, c_start, s, block_size, standardCube, swapStokesFreqAxis);

            // Write the standard dataset
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            
            std::vector<hsize_t> count = trimAxes({1, n_channels, height, width}, N); // ok as in original SmartConverter.cc it is trimAxes({1, sliceIncrement, height, width}, N);
            std::vector<hsize_t> memDims = {n_channels, height, width};
            std::vector<hsize_t> start = trimAxes({s, c_start, 0, 0}, N); // std::vector<hsize_t> start = trimAxes({s, c, 0, 0}, N);                       
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;
                    
            for(hsize_t c = c_start; c < c_end; c++) {                 
                PROGRESS_DECIMATED(c, channelProgressStride, "|");
                // read one channel
                DEBUG(std::cout << "+ Processing channel " << c << "... " << std::flush;);
                DEBUG(std::cout << " Accumulating XY stats and mipmaps..." << std::flush;);
                TIMER(timer.start(timerLabelStatsMipmaps););

                auto indexXY = c;
                hsize_t block_pos = (c-c_start)* image_size; 

                auto start1 = std::chrono::high_resolution_clock::now();
                // calculate statistics in channel indexXY  
                // TODO : check if I can just use c without duplicating variables !
                calculateChannelStats(indexXY, block_pos);

                // calculate channel histograms:
                calculateChannelHistogram(indexXY, block_pos);
            
                // Accumulate XYZ statistics
                if (depth > 1) {
                    DEBUG(std::cout << " Accumulating XYZ stats..." << std::flush;);
                    statsXY.accumulateStatsToCounter(counterXYZ, indexXY);
                }
            
                // Final mipmap calculation
                DEBUG(std::cout << " Final mipmaps..." << std::flush;);
                mipMaps.calculate();
                // add everything to processing time of the 1st pass:
                auto end1 = std::chrono::high_resolution_clock::now();
                auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
                total_first_pass_processing_ms += double(duration1.count());
            
            
                // Write the mipmaps
                auto start_io = std::chrono::high_resolution_clock::now();
                DEBUG(std::cout << " Writing mipmaps..." << std::flush;);
                TIMER(timer.start("Write"););
                mipMaps.write(s, c);
                auto end_io = std::chrono::high_resolution_clock::now();
                auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                total_io_ms += double(duration_io.count());
            
                start1 = std::chrono::high_resolution_clock::now();
                // Reset mipmaps before next channel
                DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
                TIMER(timer.start(timerLabelStatsMipmaps););
                mipMaps.resetBuffers();
                end1 = std::chrono::high_resolution_clock::now();
                duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
                total_first_pass_processing_ms += double(duration1.count());            
            } // end of first channel loop
        } // end of loop over blocks

        auto start_io = std::chrono::high_resolution_clock::now();
        // write channel stats to HDF5 files:
        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats" << std::endl);
        statsXY.write({1, depth}, {s, 0});
        auto end_io = std::chrono::high_resolution_clock::now();
        auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
        total_io_ms += double(duration_io.count());


        std::cout << "BENCHMARKING : total pure-processing time of 1st pass: " << total_first_pass_processing_ms << " milliseconds " << float(total_first_pass_processing_ms)/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += total_first_pass_processing_ms;
                
        PROGRESS(std::endl);
        
        auto start1 = std::chrono::high_resolution_clock::now();
        if (depth > 1) {
            // Final correction of XYZ min and max
            DEBUG(std::cout << " Final XYZ stats..." << std::flush;);
            PROGRESS("\tXYZ stats" << std::endl);
            TIMER(timer.start(timerLabelStatsMipmaps););
            
            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);            
            savedCubeMin[s] = statsXYZ.minVals[0];
            savedCubeMax[s] = statsXYZ.maxVals[0];
        }    

        // XY and XYZ histograms
        // We need a second pass over all channels because we need cube min and max (and channel min and max per channel)
        // We do the second pass backwards to take advantage of caching
        DEBUG(std::cout << " Histograms..." << std::endl;);
        PROGRESS("\tHistograms\t");
        TIMER(timer.start("Histograms"););
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
        total_first_pass_processing_ms += double(duration1.count());
        

        // calculate cube (statsXYZ) histogram:
//        auto start2 = std::chrono::high_resolution_clock::now();        
//        statsXYZ.clearHistogramBuffers();
//        double total_second_pass_processing_ms = 0.00;
/*        if( bApproximateCubeHistogram ) {
           // calculate approximate cube histogram by merging channel histograms
           // which is perfectly enough for selecting the right colour scale
           total_second_pass_processing_ms = calcApproxCubeHistogram(s);
        } else {
           // calculate exect cube histogram by doing second pass 
           // thought the data (slower)
           total_second_pass_processing_ms = doSecondPass(s, n_blocks, sliceIncrement, leftOverSlices, total_io_ms);
        }*/

/*        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);        
        std::cout << "Execution of 2nd big standard conversion loop over all channels took: " << duration2.count() << " milliseconds." << std::endl;
        std::cout << "BENCHMARKING : total pure-processing time of 2nd pass: " << total_second_pass_processing_ms << " milliseconds " << (float(total_second_pass_processing_ms)/1000.00) << " seconds" << std::endl;
        total_pureprocessing_ms += total_second_pass_processing_ms;*/

        
        PROGRESS(std::endl);
        
        start_io = std::chrono::high_resolution_clock::now();
        if (depth > 1) {
            // statsXYZ.write({1}, {s});
            auto basicN = statsXYZ.basicDatasetDims.size();
            statsXYZ.writeBasic(statsXYZ.fullBasicBufferDims, trimAxes({1}, basicN), trimAxes({s}, basicN));
        }
        end_io = std::chrono::high_resolution_clock::now();
        duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
        total_io_ms += double(duration_io.count());
    } // end of stokes
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
    thread_mipmaps_array.clear(); // Free thread-local buffers

// STILL TODO :             
// Rotation is performed on HDF5 file - seems to be easier this way, but there is still room for some optimisations / paralleisations:            
// Probably rotation can also be done in the first loop, why not ?
    // Swizzle
    if (depth > 1) {
        // double total_rotation_pass_processing_ms = calculateRotatedData(total_io_ms);
         double total_rotation_pass_processing_ms =
             calculateRotatedDataAndCubeHistogram(total_io_ms, savedCubeMin, savedCubeMax);
        total_pureprocessing_ms += total_rotation_pass_processing_ms;
    }
    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;
    std::cout << "BENCHMARKING : total pure-processing time of 1st, 2nd and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of entire SmartConverter::copyAndCalculate took: " << duration.count() << " milliseconds " << (float(duration.count())/1000.00) << " seconds" << std::endl;
}


double SmartConverter::calculateChannelStats( hsize_t indexXY, hsize_t block_pos )
{
   StatsCounter counterXY;

   const hsize_t REGION_MULTIPLIER = 32;
   int regionRows = std::ceil((float)height / (float)REGION_MULTIPLIER);
   int regionCols = std::ceil((float)width / (float)REGION_MULTIPLIER);
   int cubeSizeInRegions = regionRows * regionCols;

   auto start1 = std::chrono::high_resolution_clock::now();

   // We don't need to pass the array; it's a class member.
   // Just ensure OpenMP knows it's shared.
   #pragma omp parallel num_threads(allowed_mipmaps_threads) default(none) shared(std::cout, block_pos, standardDims, tileDims, zMips, standardCube, cubeSizeInRegions, mipMaps, counterXY, REGION_MULTIPLIER, width, height, thread_mipmaps_array)
   {
        int tid = 0;
        #ifdef _OPENMP
            tid = omp_get_thread_num();
        #endif

        // Grab this thread's pre-allocated MipMap object from the class member
        MipMaps& my_mipmaps = thread_mipmaps_array[tid];
        my_mipmaps.resetBuffers();
        
        StatsCounter counterRegion;
        counterRegion.reset();
        
        #pragma omp for
        for (int regionIndex = 0; regionIndex < cubeSizeInRegions; regionIndex += 1 ) {
            // ... exact same inner loop as before ...
            hsize_t x0,y0,z0;
            RegionIndexToXYZ(regionIndex, x0, y0, z0, width, height, REGION_MULTIPLIER, REGION_MULTIPLIER, 1);
            for (hsize_t y = y0; (y < y0 + REGION_MULTIPLIER && y < height); y++) {
                auto y_pos = block_pos + y * width;
                for (hsize_t x = x0; (x < x0 + REGION_MULTIPLIER && x < width); x++) {
                    auto pos = y_pos + x;
                    auto& val = standardCube[pos];
                    if (std::isfinite(val)) {
                        counterRegion.accumulateFinite(val);
                        my_mipmaps.accumulate(val, x, y, 0); 
                    } else {
                        counterRegion.accumulateNonFinite();
                    }
                }
            }
        } 
        
        #pragma omp critical
        {
            counterXY.accumulateFromCounter(counterRegion); 
            mipMaps.accumulateFromMipMaps(my_mipmaps);
        }                
   } 

   statsXY.copyStatsFromCounter(indexXY, height * width, counterXY);

   auto end1 = std::chrono::high_resolution_clock::now();
   auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
   std::cout << "Execution of 1st loop took: " << duration1.count() << " milliseconds." << std::endl;

   return duration1.count();
}

double SmartConverter::calculateChannelHistogram( hsize_t indexXY, hsize_t block_pos )
{
   auto start1 = std::chrono::high_resolution_clock::now();

   // STEP4a : histograming in the 1st pass :
   // histograming channel c :            
   double chanMin = statsXY.minVals[indexXY];
   double chanMax = statsXY.maxVals[indexXY];
   double chanRange = chanMax - chanMin;
   bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);

   if( chanHist ) {
      printf("DEBUG : calculating channel histogram for channel = %ld\n",(long int)indexXY);
      auto doChannelHistogram = [&] (float val, hsize_t offset) {
         // XY histogram
         statsXY.accumulatePartialHistogram(val, chanMin, chanRange, offset);
      };

      int portions = (height / height_chunk);
      if ( (height % height_chunk) != 0 ) {
         std::cerr << "ERROR : height chunk " << height_chunk << " is not a divider of height " << height << " due to bug in the code -> aborting now" << std::endl;
         exit(0);
      }

      hsize_t y;
      for (int p=0;p<portions;p++) {
         auto start_y = (p*height_chunk);
         auto end_y   = (p+1)*height_chunk;
#pragma omp parallel for default(none) private(y) shared(standardCube, start_y, end_y, height, width, doChannelHistogram, block_pos)
         for (y = start_y; y < end_y; y++) {
            auto y_width = block_pos + y * width;
    
            for (hsize_t x = 0; x < width; x++) {
               auto pos = y_width + x;
               auto& val = standardCube[pos];
               if (std::isfinite(val)) {
                  doChannelHistogram(val, y - start_y); // filling channel histograms for y 
               }
            }
         } // end of XY loop
         statsXY.consolidateAndClearPartialHistogram(indexXY);
      }
   }else{
      printf("channel = %ld : WARNING : channel histogram not calculated !!!???\n",(long int)indexXY);
   }

   auto end1 = std::chrono::high_resolution_clock::now();
   auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
   std::cout << "Execution of 1st loop took: " << duration1.count() << " milliseconds." << std::endl;

   return duration1.count();
}



// New SmartConverter method, replacing the call to calculateRotatedData() +
// doSecondPass()/calcApproxCubeHistogram() with a single pass that folds the
// channel (XY) and cube (XYZ) histograms into the existing rotation pass --
// same principle as MicroMemoryConverter.cc's STEP 4 (see its top-of-file
// comment block). This eliminates doSecondPass's extra depth-many FITS
// channel rereads entirely.
double SmartConverter::calculateRotatedDataAndCubeHistogram(double& total_io_ms,
                                                              const std::vector<double>& savedCubeMin,
                                                              const std::vector<double>& savedCubeMax)
{
    auto start = std::chrono::high_resolution_clock::now();
    hsize_t numTiles = std::ceil(width / TILE_SIZE) * std::ceil(height / TILE_SIZE);
    const hsize_t tileProgressStride = std::max((hsize_t)1, (hsize_t)(numTiles / 100));

    DEBUG(std::cout << "Performing tiled rotation with folded cube histogram." << std::endl;);
    PROGRESS("Tiled rotation, Z stats & cube histogram" << std::endl);
    TIMER(timer.start("Allocate"););

    hsize_t sliceSize = product(trimAxes({stokes, depth, TILE_SIZE, TILE_SIZE}, N));
    std::cout << "MEMORY : sliceSize = " << sliceSize << " stokes:" << stokes << " depth: " << depth << " TILE_SIZE:" << TILE_SIZE << " N:" << N << std::endl;
    std::cout << "MEMORY (SmartConverter::calculateRotatedDataAndCubeHistogram for standardSlice and rotatedSlice ): allocating " << double(2*sliceSize*sizeof(float))/1e9 << " GB " << std::endl << std::flush;
    float* standardSlice = new float[sliceSize];
    float* rotatedSlice = new float[sliceSize];

    printf("DEBUG : before statsZ.createBuffers({%llu,%llu})\n",TILE_SIZE,TILE_SIZE);
    statsZ.createBuffers({TILE_SIZE, TILE_SIZE});

    bool doCubeHistogram = (numBins > 0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double total_rotation_pass_processing_ms = double(duration.count());;
    double total_rotation_io_ms = 0.00;

    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
        PROGRESS("\tStokes " << s << "\t");

        // Pull this stokes' cached cube min/max (pass 1's in-memory statsXYZ
        // buffer has since moved on to later stokes -- see the caller-side
        // caching this function depends on). statsXY needs no such cache:
        // its histogram is already complete and already written by the
        // caller, right after pass 1.
        auto starthist = std::chrono::high_resolution_clock::now();

        double cubeMin = 0, cubeMax = 0, cubeRange = 0;
        bool cubeHist = false;

        if (doCubeHistogram) {
            statsXYZ.clearHistogramBuffers();

            cubeMin = savedCubeMin[s];
            cubeMax = savedCubeMax[s];
            cubeRange = cubeMax - cubeMin;
            cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;

            DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);
        }
        auto endhist = std::chrono::high_resolution_clock::now();
        total_rotation_pass_processing_ms += double(std::chrono::duration_cast<std::chrono::milliseconds>(endhist - starthist).count());

        hsize_t tileCount(0);
        for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
            for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                tileCount++;
                hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
                hsize_t ySize = std::min(TILE_SIZE, height - yOffset);

                DEBUG(std::cout << "+ Processing tile slice at " << xOffset << ", " << yOffset << "..." << std::flush;);
                PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");

                // read tile slice
                auto start_io = std::chrono::high_resolution_clock::now();

                DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                TIMER(timer.start("Read"););

                auto standardMemDims = trimAxes({1, depth, ySize, xSize}, N);
                auto standardCount = trimAxes({1, depth, ySize, xSize}, N);
                auto standardStart = trimAxes({s, 0, yOffset, xOffset}, N);

                readHdf5Data(standardDataSet, standardSlice, standardMemDims, standardCount, standardStart);
                auto end_io = std::chrono::high_resolution_clock::now();
                auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                total_rotation_io_ms += double(duration_io.count());
                std::cout << "3nd I/O (readHdf5Data) for xOffset/yOffset : " << xOffset << " , " << yOffset << " took " << duration_io.count() << " milliseconds." << std::endl;

                auto starttile = std::chrono::high_resolution_clock::now();
                // rotate tile slice
                DEBUG(std::cout << " Calculating rotation..." << std::flush;);
                TIMER(timer.start("Rotation"););

                hsize_t tile_size = (ySize * xSize);
                hsize_t ysize_depth = (ySize * depth);
                hsize_t i;

                const hsize_t BLOCK_SIZE = 16;

                const size_t dest_y_stride = depth;
                const size_t dest_z_stride = depth * ySize;
                const size_t src_y_stride  = xSize;
                const size_t src_z_stride  = ySize * xSize;

                #pragma omp parallel for collapse(3) schedule(dynamic)
                for (hsize_t i0 = 0; i0 < depth; i0 += BLOCK_SIZE) {
                    for (hsize_t j0 = 0; j0 < ySize; j0 += BLOCK_SIZE) {
                        for (hsize_t k0 = 0; k0 < xSize; k0 += BLOCK_SIZE) {

                            hsize_t i_max = std::min(i0 + BLOCK_SIZE, depth);
                            hsize_t j_max = std::min(j0 + BLOCK_SIZE, ySize);
                            hsize_t k_max = std::min(k0 + BLOCK_SIZE, xSize);

                            for (hsize_t i = i0; i < i_max; ++i) {
                                for (hsize_t j = j0; j < j_max; ++j) {

                                    size_t i_plus_depth_j = i + dest_y_stride * j;
                                    size_t src_base_idx = i * src_z_stride + j * src_y_stride;

                                    #pragma omp simd
                                    for (hsize_t k = k0; k < k_max; ++k) {
                                        size_t destIndex = i_plus_depth_j + dest_z_stride * k;
                                        size_t srcIndex  = src_base_idx + k;

                                        rotatedSlice[destIndex] = standardSlice[srcIndex];
                                    }
                                }
                            }
                        }
                    }
                }

                // A separate pass over the same slice depth-last
                DEBUG(std::cout << " Calculating Z statistics..." << std::flush;);
                TIMER(timer.start("Z statistics"););

                Stats* statsZ_ptr = &statsZ;

                #pragma omp parallel default(none) shared(ySize, xSize, depth, standardSlice, statsZ_ptr, tile_size)
                {
                   StatsCounter counterZ;

                  #pragma omp for collapse(2)
                  for (hsize_t j = 0; j < ySize; j++) {
                     for (hsize_t k = 0; k < xSize; k++) {

                        counterZ.reset();
                        auto indexZ = k + xSize * j;

                        for (hsize_t i = 0; i < depth; i++) {
                           auto sourceIndex = indexZ + tile_size * i;
                           auto& val = standardSlice[sourceIndex];
                           if (std::isfinite(val)) {
                              counterZ.accumulateFinite(val);
                           } else {
                              counterZ.accumulateNonFinite();
                           }
                        }
                        statsZ_ptr->copyStatsFromCounter(indexZ, depth, counterZ);
                     }
                  }
                }

                // Fold the CUBE histogram into this same pass, on the same
                // standardSlice values already read above -- no extra I/O.
                // (statsXY's histogram is NOT touched here -- it's already
                // correct, from pass 1, and already written by the caller.)
                //
                // Deliberately SEQUENTIAL, not folded into the #pragma omp
                // block above: histogram bins are value-based, not spatial --
                // threads handling different (j,k) still share the same cube
                // bins, and Stats::accumulateHistogram's plain `++` isn't
                // atomic. The Z-statistics loop above has no such collision
                // (each (j,k) owns its own statsZ slot), so it stays parallel.
                if (doCubeHistogram && cubeHist) {
                    TIMER(timer.start("Histograms"););
                    for (hsize_t j = 0; j < ySize; j++) {
                        for (hsize_t k = 0; k < xSize; k++) {
                            for (hsize_t i = 0; i < depth; i++) {
                                auto sourceIndex = k + xSize * j + tile_size * i;
                                auto& val = standardSlice[sourceIndex];
                                if (std::isfinite(val)) {
                                    statsXYZ.accumulateHistogram(val, cubeMin, cubeRange, 0);
                                }
                            }
                        }
                    }
                }

                auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);

                auto endtile = std::chrono::high_resolution_clock::now();
                auto durationtile = std::chrono::duration_cast<std::chrono::milliseconds>(endtile - starttile);
                total_rotation_pass_processing_ms += double(durationtile.count());
                std::cout << "Pure processing of rotation of 1 tile, including writting, took: " << durationtile.count() << " milliseconds." << std::endl;

                DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                TIMER(timer.start("Write"););

                start_io = std::chrono::high_resolution_clock::now();
                writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                end_io = std::chrono::high_resolution_clock::now();
                duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                total_rotation_io_ms += double(duration_io.count());
                std::cout << "4th I/O (writeHdf5Data) for xOffset/yOffset : " << xOffset << " , " << yOffset << " took " << duration_io.count() << " milliseconds." << std::endl;

                DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
                start_io = std::chrono::high_resolution_clock::now();
                statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
                end_io = std::chrono::high_resolution_clock::now();
                duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                total_rotation_io_ms += double(duration_io.count());
                std::cout << "5th I/O (statsZ.write) for xOffset/yOffset : " << xOffset << " , " << yOffset << " took " << duration_io.count() << " milliseconds." << std::endl;
            }
        }

        // Write the cube histogram now that this stokes' tile loop has filled
        // it in. statsXY needs no write here -- it was already written in full
        // by the caller, right after pass 1.
        if (doCubeHistogram) {
            auto start_io = std::chrono::high_resolution_clock::now();
            auto basicN = statsXYZ.basicDatasetDims.size();
            auto histN = basicN + 1;
            statsXYZ.writeHistogram(statsXYZ.fullBasicBufferDims, trimAxes(extend({1}, {statsXYZ.numBins}), histN), trimAxes(extend({s}, {0}), histN));
            auto end_io = std::chrono::high_resolution_clock::now();
            total_rotation_io_ms += double(std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io).count());
        }

        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start);
        std::cout << "Execution of loop over X/Y Offsets for Stokes = " << s << " took: " << duration1.count() << " milliseconds." << std::endl;

        PROGRESS(std::endl);

    } // end of loop over Stokeses

    total_io_ms += total_rotation_io_ms;
    std::cout << "BENCHMARKING : total rotation I/O took: " << total_rotation_io_ms << " milliseconds "
               << (float(total_rotation_io_ms)/1000.0) << " seconds" << std::endl;

    auto startfree = std::chrono::high_resolution_clock::now();
    TIMER(timer.start("Free"););
    DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
    delete[] standardSlice;
    delete[] rotatedSlice;
    end = std::chrono::high_resolution_clock::now();
    auto duration_free = std::chrono::duration_cast<std::chrono::milliseconds>(end - startfree);
    total_rotation_pass_processing_ms += double(duration_free.count());
    std::cout << "BENCHMARKING : total pure-processing time of rotation pass: " << total_rotation_pass_processing_ms << " milliseconds "
              << (float(total_rotation_pass_processing_ms)/1000.0) << " seconds" << std::endl;

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double total_rotation_ms = double(duration.count());
    std::cout << "BENCHMARKING : rotation pass took: " << total_rotation_ms << " milliseconds "
              << (float(total_rotation_ms)/1000.0) << " seconds" << std::endl;

    return total_rotation_pass_processing_ms;

}

IOCostBreakdown SmartConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                            hsize_t numBins,
                                            const IOCostModel& readModel,
                                            const IOCostModel& writeModel) {
    IOCostBreakdown result;

    std::vector<hsize_t> standardDims = {stokes, depth, height, width};
    std::vector<hsize_t> chunkDims    = {1, 1, TILE_SIZE, TILE_SIZE};

    int sliceIncrement = n_io_blocks;
    int sliceIncrementCount = (int)(depth / sliceIncrement);
    int leftOverSlices = (int)(depth % sliceIncrement);
    int nBlocks = sliceIncrementCount + (leftOverSlices > 0 ? 1 : 0);

    // ---------- 1st pass: block read/write ----------
    {
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            hsize_t bytesPerOp = nChannels * height * width * sizeof(float);
            acc.add(repeatEstimate(IOOpEstimate{1, bytesPerOp, bytesPerOp}, stokes), readModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: FITS block read (readFitsData)"));
    }
    {
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            auto perBlock = estimateHyperslabIO(standardDims, chunkDims,
                                                 {1, nChannels, height, width}, sizeof(float));
            acc.add(repeatEstimate(perBlock, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet block write"));
    }

    // calculateChannelStats / calculateChannelHistogram: pure in-memory compute,
    // zero I/O -- channel histogram fully finalized here, not deferred.

    {
        PhaseAccumulator acc;
        hsize_t mipXY = 1, hLevel = height, wLevel = width;
        do {
            mipXY *= 2;
            hLevel = (hsize_t)std::ceil((double)height / mipXY);
            wLevel = (hsize_t)std::ceil((double)width  / mipXY);
            IOOpEstimate perLevel{ 1, hLevel * wLevel * sizeof(double), hLevel * wLevel * sizeof(double) };
            acc.add(repeatEstimate(perLevel, stokes * depth), writeModel);
        } while (2 * wLevel > MIN_MIPMAP_SIZE || 2 * hLevel > MIN_MIPMAP_SIZE);
        result.phases.push_back(acc.toPhase("1st pass: mipmap writes (all levels)"));
    }

    // statsXY.write(...): ONE combined call, basic + histogram together.
    {
        PhaseAccumulator acc;
        hsize_t elemSizes[] = {4, 4, 8, 8, 8};
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, 8);
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: statsXY write (basic + histogram)"));
    }

    // statsXYZ.writeBasic(...): basic only -- cube histogram deferred to rotation.
    if (depth > 1) {
        PhaseAccumulator acc;
        hsize_t elemSizes[] = {4, 4, 8, 8, 8};
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: statsXYZ writeBasic"));
    }

    // ---------- Rotation pass (only when depth > 1) ----------
    addTiledRotationPhases(result, stokes, depth, height, width, readModel, writeModel);

    // statsXYZ.writeHistogram(...): the cube histogram, once per stokes, after
    // the tile loop fills it in.
    if (depth > 1 && numBins > 0) {
        PhaseAccumulator acc;
        auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, 8);
        acc.add(repeatEstimate(h, stokes), writeModel);
        result.phases.push_back(acc.toPhase("Rotation pass: statsXYZ writeHistogram"));
    }

    return result;
}