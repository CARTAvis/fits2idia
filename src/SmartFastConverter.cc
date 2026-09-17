/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"
#include <algorithm> // for std::min

SmartFastConverter::SmartFastConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) 
 : SmartConverter(inputFileName, outputFileName, progress, zMips)
{}

MemoryUsage SmartFastConverter::calculateMemoryUsage() {
    MemoryUsage m;
    
    std::cout << "MEMORY_ESTIMATE: (SmartConverter::calculateMemoryUsage) parameters: n_io_blocks = " << n_io_blocks << " , height = " << height << " , width = " << width << std::endl;

    // memory used in pass 1 :
    m.sizes["Main dataset"] = n_io_blocks * height * width * sizeof(float); // multiple (_sliceIncrement) image slices can be read in 1 block 
    std::cout << "MEMORY_ESTIMATE: Main dataset  = " << m.sizes["Main dataset"]* 1e-9 << " GB" << std::endl;
    
    m.sizes["XY stats"] = Stats::size({depth}, numBins); // statsXY.createBuffers({depth});
    std::cout << "MEMORY_ESTIMATE: XY stats  = " << m.sizes["XY stats"]* 1e-9 << " GB" << std::endl;
    
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {(hsize_t)n_io_blocks, height, width}, zMips);
    std::cout << "MEMORY_ESTIMATE: MipMaps calculations = " << m.sizes["Mipmaps"] * 1e-9 << " GB " << std::endl;

    // DONE up to here:
    if (depth > 1) {
       // rotated:
       m.sizes["Rotation"] = n_io_blocks * height * width * sizeof(float); // multiple (_sliceIncrement) image slices can be read in 1 block 
       std::cout << "MEMORY_ESTIMATE: Rotated dataset  = " << m.sizes["Rotation"]* 1e-9 << " GB" << std::endl;
    
       m.sizes["XYZ stats"] = Stats::size({}, numBins, n_io_blocks ); // has to match : statsXYZ.createBuffers({}, sliceIncrement);
       std::cout << "MEMORY_ESTIMATE: XYZ stats  = " << m.sizes["XYZ stats"]* 1e-9 << " GB" << std::endl;
       
       m.sizes["Z stats"] = Stats::size({height, width}); // statsZ.createBuffers({height, width});
       std::cout << "MEMORY_ESTIMATE: Z stats  = " << m.sizes["Z stats"]* 1e-9 << " GB , for numBins = " << numBins << std::endl;
       
       m.sizes["globalCountersZ"] = height  * width * sizeof(StatsCounter);
       std::cout << "MEMORY_ESTIMATE: globalCountersZ  = " << m.sizes["globalCountersZ"]* 1e-9 << " GB" << std::endl;
    }

    hsize_t total_pass1 = 0;
    for (auto& kv : m.sizes) {
        total_pass1 += kv.second;
    }
       
    std::cout << "MEMORY_ESTIMATE: 1st pass = " << total_pass1 * 1e-9 << " GB " << std::endl;
    //----------------------------------------------- end of 1st pass -----------------------------------------------

    // second pass :    
    hsize_t total_pass2 = 0;
    std::cout << "MEMORY_ESTIMATE: 1st pass = " << total_pass2 * 1e-9 << " GB " << std::endl;

    m.total = std::max(total_pass1, total_pass2);
    std::cout << "MEMORY_ESTIMATE: peak usage = " << m.total * 1e-9 << " GB " << std::endl;

    return m;
}

void SmartFastConverter::copyAndCalculate() {
    const hsize_t pixelProgressStride = std::max((hsize_t)1, (hsize_t)(width * height / 100));

    auto start = std::chrono::high_resolution_clock::now();
    
    if(memoryLimitInMb <= 0) {
       std::cerr << "ERROR : memory limit not set for the SmartConverter and it is strictly required -> exiting SmartConverter::copyAndCalculate function" << std::endl;
       return;
    }
    
    // 
    int sliceIncrement = n_io_blocks;
    int sliceIncrementCount = depth/sliceIncrement;
    int leftOverSlices = (depth % sliceIncrement);    
    int n_blocks = sliceIncrementCount;
    if (leftOverSlices > 0) {
       n_blocks++;
    }
    std::cout << "DEBUG : n_blocks = " << n_blocks << " vs.  sliceIncrementCount = " << sliceIncrementCount << " and leftOverSlices = " << leftOverSlices << std::endl;
              
              
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    
    // Allocate one channel at a time, and no swizzled data
    TIMER(timer.start("Allocate"););
    std::cout << "MEMORY : stokes:" << stokes << " depth: " << depth << " TILE_SIZE:" << TILE_SIZE << " N:" << N << std::endl;
    std::cout << "MEMORY (SmartConverter::copyAndCalculate): allocating standardCube with size " << double(height * width * sliceIncrement*sizeof(float))/1e9 << " GB " << std::endl << std::flush;
    standardCube = new float[height * width * sliceIncrement];    
    if (depth > 1) {
       std::cout << "MEMORY (SmartConverter::copyAndCalculate): allocating rotatedCube with size " << double(height * width * sliceIncrement*sizeof(float))/1e9 << " GB " << std::endl << std::flush;
       rotatedCube = new float[height * width * sliceIncrement];
    }
    
    
    // Allocate one stokes of stats at a time
    // statsXY.createBuffers({depth});
    std::cout << "DEBUG : before statsXY.createBuffers({" << depth << ")" << std::endl;
//    statsXY.createBuffers({depth}, height_chunk);
    statsXY.createBuffers({depth});
        
    if (depth > 1) {
        std::cout << "DEBUG : before statsXYZ.createBuffers({}," << sliceIncrement << ")" << std::endl;
        // Allocate based on the maximum channels per block (c - c_start)
        statsXYZ.createBuffers({}, sliceIncrement);

        statsZ.createBuffers({height, width});
        std::cout << "DEBUG : before statsZ.createBuffers({" << height << "," << width << "})" << std::endl;
    }

    // Change the depth from 1 to sliceIncrement
    printf("DEBUG : before mipMaps.createBuffers({%d,%llu,%llu})\n", sliceIncrement, height, width);    
    mipMaps.createBuffers({(hsize_t)sliceIncrement, height, width});

    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";

    hsize_t image_size = width*height;

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    for (unsigned int s = 0; s < stokes; s++) {
        unsigned int currentStokes = s;
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        
        PROGRESS("\tMain loop\t");
        
        // ADD THIS: Clear channel histogram accumulation buffers ONCE per Stokes
        statsXY.clearHistogramBuffers();
        
        // ADD THIS: Persistent counters that survive across Z-axis blocks
        std::vector<StatsCounter> globalCountersZ;
        if (depth > 1) {
            std::cout << "MEMORY (SmartConverter::copyAndCalculate): allocating globalCountersZ with size " << double(height * width * sizeof(StatsCounter))/1e9 << " GB " << std::endl << std::flush;
            globalCountersZ.resize(width * height);
        }
        
        double total_first_pass_processing_ms = 0.00;
        for (hsize_t block = 0; block < n_blocks; block++ ) {
            hsize_t c_start = block*sliceIncrement;
            hsize_t c_end   = c_start + sliceIncrement;
            if (block == (n_blocks-1) && leftOverSlices > 0 ) {
               c_end   = c_start + leftOverSlices; // if last block and there are left over slices                
            }
            hsize_t n_channels = (c_end-c_start);
            hsize_t block_size = n_channels*height*width;
            
            // Resize the mipMaps buffer for the final block so it doesn't write out-of-bounds
            // dimensions of MipMaps have to be adjusted for smaller number of channels in the last porition of the data:
            if (block == (n_blocks - 1) && leftOverSlices > 0) {
                mipMaps.createBuffers({n_channels, height, width});
            }
            
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
            auto block_io_ms = double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;

                    
            auto start_processing = std::chrono::high_resolution_clock::now();
            std::cout << "PROGRESS : before statsXY accumulation ..." << std::endl;
            #pragma omp parallel for
            for (hsize_t i = c_start; i < c_end; i++) {
                PROGRESS_DECIMATED(i, channelProgressStride, "|");
                
                StatsCounter counterXY;
                counterXY.reset(); // CRITICAL: Clear dirty thread-stack memory
                
                for (hsize_t j = 0; j < height; j++) {
                    for (hsize_t k = 0; k < width; k++) {
                        auto sourceIndex = k + width * j + (height * width) * (i - c_start);
                        // auto destIndex = (i-c_start) + sliceIncrement * j + (height * sliceIncrement) * k;
                        // Use n_channels instead of sliceIncrement to pack memory correctly:
                        auto destIndex = (i-c_start) + n_channels * j + (height * n_channels) * k;
                        auto& val = standardCube[sourceIndex];
                        
                        if (depth > 1) {
                            rotatedCube[destIndex] = val;
                        }
                        
                        // Accumulate XY stats matching the OLD code exactly
                        if (std::isfinite(val)) {
                            counterXY.accumulateFinite(val);
                        } else {
                            counterXY.accumulateNonFinite();
                        }
                    }
                }
                
                // Final correction of XY min and max
                statsXY.copyStatsFromCounter(i, height * width, counterXY);
            }
            std::cout << "PROGRESS : after statsXY accumulation ..." << std::endl;
            PROGRESS(std::endl);

            // -------------------------------------------------------------
            // ADD THIS: Accumulate Z statistics for the current block
            // -------------------------------------------------------------
            if (depth > 1) {
                DEBUG(std::cout << " Z statistics accumulation for block..." << std::flush;);
                
                // Swapped order of the loop so that the loop over k (image X-axis) is last to optmise L2 cache
                #pragma omp parallel for
                for (hsize_t j = 0; j < height; j++) {
                // Swap 'i' to the middle
                for (hsize_t i = c_start; i < c_end; i++) {
            
                    // Make 'k' the innermost loop for contiguous memory access
                    for (hsize_t k = 0; k < width; k++) {
                        auto indexZ = k + j * width;
                        auto sourceIndex = k + width * j + (height * width) * (i - c_start);
                        auto& val = standardCube[sourceIndex];

                        // Safely fetch and update the counter for this specific pixel
                        auto& counterZ = globalCountersZ[indexZ];
                
                        if (std::isfinite(val)) {
                            counterZ.accumulateFinite(val);
                        } else {
                            counterZ.accumulateNonFinite();
                        }
                    }
                }
                }
            }


            // NEW CODE:
            // New Chunked Channel Histogram Generation Loop
            DEBUG(std::cout << " Channel Histograms..." << std::flush;);
            PROGRESS("\tChannel Histograms\t");
            TIMER(timer.start("Histograms"););

#pragma omp parallel for
            for (hsize_t i = c_start; i < c_end; i++) {
                PROGRESS_DECIMATED(i, channelProgressStride, "|");
                
                auto& indexXY = i;
                double chanMin = statsXY.minVals[indexXY];
                double chanMax = statsXY.maxVals[indexXY];
                double chanRange = chanMax - chanMin;
                
                bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
                
                if (!chanHist) {
                    continue; 
                }

                for (hsize_t j = 0; j < width * height; j++) {
                    // Fetch index relative to current block memory footprint
                    auto sourceIndex = (i - c_start) * width * height + j;
                    auto& val = standardCube[sourceIndex];

                    if (std::isfinite(val)) {
                        statsXY.accumulateHistogram(val, chanMin, chanRange, i);
                    }
                } 
            }                     
            auto end_processing = std::chrono::high_resolution_clock::now();
            auto duration_processing = std::chrono::duration_cast<std::chrono::milliseconds>(end_processing - start_processing);
            auto block_first_pass_processing_ms = double(duration_processing.count());
            

            DEBUG(std::cout << " Writing main and rotated datasets... " << std::flush;);
            PROGRESS("\tWrite data" << std::endl);
            TIMER(timer.start("Write"););
                 
/*            start_io = std::chrono::high_resolution_clock::now();                        
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            block_io_ms += double(duration_io.count());
            std::cout << "I/O writeHdf5Data(standardDataSet) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;*/
        
            if (depth > 1) {
                start_io = std::chrono::high_resolution_clock::now();
                // This all technically worked if we reused the standard filespace and memspace
                // But it's probably not a good idea to rely on two incorrect values cancelling each other out
                // Use n_channels instead of depth            
                std::vector<hsize_t> swizzledCount = trimAxes({1, width, height, n_channels}, N);
                std::vector<hsize_t> swizzledMemDims = {width, height, n_channels};            
                // Standard format is {Stokes, Depth, Height, Width} -> Start is {s, c_start, 0, 0}
                // Swizzled format is {Stokes, Width, Height, Depth} -> Start is {s, 0, 0, c_start}
                std::vector<hsize_t> swizzledStart = trimAxes({currentStokes, 0, 0, c_start}, N);
                
                writeHdf5Data(swizzledDataSet, rotatedCube, swizzledMemDims, swizzledCount, swizzledStart);
                
                end_io = std::chrono::high_resolution_clock::now();
                duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
                block_io_ms += double(duration_io.count());
                std::cout << "I/O writeHdf5Data(swizzledDataSet) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;
            }

            // Fourth loop handles mipmaps        
            // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
            DEBUG(std::cout << " Mipmaps..." << std::endl;);
            PROGRESS("\tMipmaps\t\t");
            TIMER(timer.start("Mipmaps"););


            start_processing = std::chrono::high_resolution_clock::now();                
#pragma omp parallel for
            for (hsize_t c = c_start; c < c_end; c++) {
                PROGRESS_DECIMATED(c, channelProgressStride, "|");
                for (hsize_t y = 0; y < height; y++) {
                    for (hsize_t x = 0; x < width; x++) {
                        auto sourceIndex = x + width * y + (height * width) * (c - c_start);
                        auto& val = standardCube[sourceIndex];
                        if (std::isfinite(val)) {
                            mipMaps.accumulate(val, x, y, c - c_start);
                        }
                    }
                }
            } // end of mipmap loop
            PROGRESS(std::endl);
        
            // Final mipmap calculation
            mipMaps.calculate();
            std::cout << "PROGRESS : after mipMaps accumulation ..." << std::endl;            
            end_processing = std::chrono::high_resolution_clock::now();
            duration_processing = std::chrono::duration_cast<std::chrono::milliseconds>(end_processing - start_processing);
            block_first_pass_processing_ms += double(duration_processing.count());

            
            TIMER(timer.start("Write"););
            PROGRESS("\tWrite stats & mipmaps" << std::endl);
            
            // Write the mipmaps
            // FIX: Pass c_start so the HDF5 writer knows the correct Z-offset
            start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(currentStokes, c_start);        
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            block_io_ms += double(duration_io.count());
            std::cout << "I/O (mipMaps.write) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;
            
            
            // Write the statistics                
            // Clear the mipmaps before the next BLOCK (not Stokes)
            TIMER(timer.start("Mipmaps"););
            mipMaps.resetBuffers();
            
            total_io_ms += block_io_ms;
            total_first_pass_processing_ms += block_first_pass_processing_ms;
            std::cout << "BENCHMARKING : block = " << block << " I/O took " << block_io_ms/1000.00 << " sec -> total I/O time = " << total_io_ms/1000.00 << " sec." << std::endl;
            std::cout << "BENCHMARKING : block = " << block << " pure processing took " << block_first_pass_processing_ms/1000.00 << " sec -> total pure processing took " << total_first_pass_processing_ms/1000.00 << " sec." << std::endl;
            
        } // end of loop over blocks
        
        // === POST-BLOCK GLOBAL CONSOLIDATION ===
        if (depth > 1) {
            std::cout << "Finalizing Global XYZ and Z statistics..." << std::endl;

            auto start_processing = std::chrono::high_resolution_clock::now();            
            // 1. Determine actual global min/max by scanning completed channel stats
            StatsCounter counterXYZ;
            for (hsize_t i = 0; i < depth; i++) {
                statsXY.accumulateStatsToCounter(counterXYZ, i);
            }
            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);
            
            std::cout << "DEBUG: copied statsXYZ from counter ..." << std::endl;

            // 2. Finalize Z-stats from persistent cross-block buffers
            #pragma omp parallel for
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    auto indexZ = k + width * j;
                    statsZ.copyStatsFromCounter(indexZ, depth, globalCountersZ[indexZ]);
                }
            }
            auto end_processing = std::chrono::high_resolution_clock::now();
            auto duration_processing = std::chrono::duration_cast<std::chrono::milliseconds>(end_processing - start_processing);
            total_first_pass_processing_ms += double(duration_processing.count());
            std::cout << "BENCHMARKING : total pure-processing time of 1st pass: " << total_first_pass_processing_ms << " milliseconds " << (float(total_first_pass_processing_ms)/1000.00) << " seconds" << std::endl;
            total_pureprocessing_ms += total_first_pass_processing_ms;

            std::cout << "DEBUG: copied statsZ from counter ..." << std::endl;
            
            // 3. Compute the approximate cube histogram using your built-in algorithm
            // This populates the statsXYZ histogram buffers BEFORE we write to disk.
            auto start2 = std::chrono::high_resolution_clock::now();
            double total_second_pass_processing_ms = 0.00;
            if( bApproximateCubeHistogram ) {
               // calculate approximate cube histogram by merging channel histograms
               // which is perfectly enough for selecting the right colour scale
               total_second_pass_processing_ms = calcApproxCubeHistogram(s);
            } else {
               // calculate exect cube histogram by doing second pass 
               // thought the data (slower)
               total_second_pass_processing_ms = doSecondPass(s, n_blocks, sliceIncrement, leftOverSlices, total_io_ms);
            }

            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
            std::cout << "Execution of 2nd big standard conversion loop over all channels took: " << duration2.count() << " milliseconds." << std::endl;
            std::cout << "BENCHMARKING : total pure-processing time of 2nd pass: " << total_second_pass_processing_ms << " milliseconds " << (float(total_second_pass_processing_ms)/1000.00) << " seconds" << std::endl;
            total_pureprocessing_ms += total_second_pass_processing_ms;
            
            // 4. Write Z stats and the fully completed Global XYZ object
            auto start_io = std::chrono::high_resolution_clock::now();
            statsZ.write({height, width}, {1, height, width}, {currentStokes, 0, 0});
            statsXYZ.write({1}, {currentStokes});
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
        }
        
        auto start_io = std::chrono::high_resolution_clock::now();
        // 5. Write completed XY channel stats
        statsXY.write({1, depth}, {currentStokes, 0});
        auto end_io = std::chrono::high_resolution_clock::now();
        auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
        total_io_ms += double(duration_io.count());
    } // end of stokes
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    auto start_processing = std::chrono::high_resolution_clock::now();
    delete[] standardCube;
    if (depth > 1) {               // ADD THIS
        delete[] rotatedCube;      // ADD THIS
    }
    auto end_processing = std::chrono::high_resolution_clock::now();
    auto duration_processing = std::chrono::duration_cast<std::chrono::milliseconds>(end_processing - start_processing);
    total_pureprocessing_ms += double(duration_processing.count());

    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;
    std::cout << "BENCHMARKING : total pure-processing time of 1st, 2nd and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of entire SmartConverter::copyAndCalculate took: " << duration.count() << " milliseconds " << (float(duration.count())/1000.00) << " seconds" << std::endl;
}

double SmartFastConverter::calcApproxCubeHistogram( unsigned int s ) {
   auto start = std::chrono::high_resolution_clock::now();
   printf("INFO : SmartConverter::calcApproxCubeHistogram\n");
   // calculate cubeMin / cubeMax from minVals/maxVals in statsXY 
   const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));

   hsize_t cubeSize = height * width;
   double cubeMin;
   double cubeMax;
   double cubeRange;
   double cubeBinWidth;
   bool cubeHist(false);

   double total_second_pass_processing_ms = 0.0;
   auto start1 = std::chrono::high_resolution_clock::now();

   if (depth > 1) {
      // only create cube (XYZ) histogram if there is more than 1 channel
      // otherwise cube histogram = channel histogram
      cubeMin = statsXYZ.minVals[0];
      cubeMax = statsXYZ.maxVals[0];
      cubeRange = cubeMax - cubeMin;
      cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
      cubeBinWidth = cubeRange / numBins;
   }

   
   printf("DEBUG : cubeMin = %.8f , cubeMax = %.8f -> cubeRange = %.8f , cubeHist = %d\n",cubeMin,cubeMax,cubeRange,cubeHist);   
   DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);

   if(!cubeHist) {
      return -1;
   }        

   if (cubeHist) {
      statsXYZ.clearHistogramBuffers();

      // only a single histogram per cube (no need for offsets etc):
      double* histogram_XYZ = new double[numBins];      
      memset(histogram_XYZ,'\0', sizeof(double)*numBins);
      std::cout << "MEMORY (SmartConverter::calcApproxCubeHistogram): allocating " << double(numBins*sizeof(double))/1e9 << " GB " << std::endl << std::flush;

      auto addFractionalCount = [&] (int binIndex, float val) {
         histogram_XYZ[binIndex] += val;
      };

      long int totalBinCount = 0;        
      // 2. Iterate through every channel's histogram
      for (hsize_t c = 0; c < depth; ++c) {
         double chanMin = statsXY.minVals[c];
         double chanMax = statsXY.maxVals[c];
         double chanRange = chanMax - chanMin;
            
         if (!std::isfinite(chanMin) || !std::isfinite(chanMax) || chanRange <= 0) {
             continue;
         }

         // Calculate the width of a single bin for this specific channel
         double chanBinWidth = chanRange / numBins; 

         // 3. Iterate through the bins of the current channel
         for (size_t b = 0; b < numBins; ++b) {
            // NOTE: You will need to expose or access the actual histogram array from statsXY here
            uint64_t binCount = statsXY.getBinCount(c, b); 
            totalBinCount += binCount;
            // printf("DEBUG: c = %d, b = %d -> binCount = %d\n",int(c),int(b),int(binCount));
                
            if (binCount > 0) {
               // Assume the values in this bin are concentrated at the bin's center
               double binCenterValue = chanMin + (b * chanBinWidth) + (chanBinWidth / 2.0);

               // 4. Map the center value to the corresponding bin in the global cube histogram
               // Note: You will need to write a method like addCountToHistogram that takes a weight/count
               // statsXYZ.accumulateHistogram(binCenterValue, cubeMin, cubeRange, 0, binCount); 

               // 1. Define the exact boundaries of the source (channel) bin
               double srcStart = chanMin + (b * chanBinWidth);
               double srcEnd   = srcStart + chanBinWidth;

               // 2. Map those boundaries to exact, floating-point bin indices in the global histogram
               double destStartFloat = (srcStart - cubeMin) / cubeBinWidth;
               double destEndFloat   = (srcEnd - cubeMin) / cubeBinWidth;

               // 3. Find the integer index of the first and last destination bins touched
               // (Using std::clamp or manual bounds checking to ensure we don't write out of bounds)
               int firstDestBin = std::max(0, (int)std::floor(destStartFloat));
               int lastDestBin  = std::min((int)numBins - 1, (int)std::floor(destEndFloat));

               // 4. Distribute the counts
               if (firstDestBin == lastDestBin) {
                  // The source bin fits entirely inside a single destination bin
                  addFractionalCount(firstDestBin, (double)binCount); 
               } else {
                  // The source bin spans two (or more) destination bins!
        
                  // Count for the first bin (from srcStart to the boundary of the next dest bin)
                  double firstBinUpperBoundary = cubeMin + (firstDestBin + 1) * cubeBinWidth;
                  double firstBinFraction = (firstBinUpperBoundary - srcStart) / chanBinWidth;
                  addFractionalCount(firstDestBin, binCount * firstBinFraction);

                 // Count for the last bin (from the boundary of the last dest bin to srcEnd)
                 double lastBinLowerBoundary = cubeMin + lastDestBin * cubeBinWidth;
                 double lastBinFraction = (srcEnd - lastBinLowerBoundary) / chanBinWidth;
                 addFractionalCount(lastDestBin, binCount * lastBinFraction);

                 // Count for any full bins entirely engulfed in the middle, when cubeBinWidth << chanBinWidth (cubeBinWidth < 2*cubeBinWidth)             
                 // (Rare, but possible if the channel range is much wider than the global range)
                 double middleBinFraction = cubeBinWidth / chanBinWidth; // <1 because cubeBinWidth < chanBinWidth
                 for (int middleBin = firstDestBin + 1; middleBin < lastDestBin; ++middleBin) {
                    addFractionalCount(middleBin, binCount * middleBinFraction);
                 }
               }
            }
        }
      }
      printf("DEBUG : totalBinCount Pre-merged = %ld\n",totalBinCount);

      // TODO : copy from histogram_XYZ -> statsXYZ.histograms 
      double totalBinCountMerged = 0.00;
      for(size_t b = 0; b < numBins; ++b) {
         // WARNING : truncation of fractional parts is required here:
         double binStart = cubeMin + (b * cubeBinWidth);
         double binEnd   = binStart + cubeBinWidth;
         int64_t finalBinValue = int64_t(round(histogram_XYZ[b]));
         printf("DEBUG : calcApproxCubeHistogram : N(%d, %.6f - %.6f) = %.6f -> %ld\n",int(b),binStart,binEnd,histogram_XYZ[b],(long int)finalBinValue);
         statsXYZ.setBinCount(0, b, finalBinValue);

         totalBinCountMerged += histogram_XYZ[b];
      }
      printf("DEBUG : totalBinCountMerged = %.8f\n",totalBinCountMerged);

      delete [] histogram_XYZ;
   }
   auto end1 = std::chrono::high_resolution_clock::now();
   auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
   total_second_pass_processing_ms = double(duration1.count());
   
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start);
   std::cout << "Execution of Approx histogram calculation took: " << duration.count() << " milliseconds." << std::endl;

   return total_second_pass_processing_ms;
}


double SmartFastConverter::doSecondPass( unsigned int s, int n_blocks, int sliceIncrement, int leftOverSlices, double& total_io_ms )
{
    auto start = std::chrono::high_resolution_clock::now();
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    
    hsize_t cubeSize = height * width;
    double cubeMin;
    double cubeMax;
    double cubeRange;
    double cubeBinWidth;
    bool cubeHist(false);

    if (depth > 1) {
        cubeMin = statsXYZ.minVals[0];
        cubeMax = statsXYZ.maxVals[0];
        cubeRange = cubeMax - cubeMin;
        cubeBinWidth = cubeRange / numBins;
        cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
    }

    if(!cubeHist) {
       return -1;
    }        

    statsXYZ.clearHistogramBuffers();
    double total_second_pass_processing_ms = 0.00;
    auto start1 = std::chrono::high_resolution_clock::now();

    // Loop through blocks forward-aligned to memory layout
    for (int block = 0; block < n_blocks; block++) {
        hsize_t c_start = block * sliceIncrement;
        hsize_t c_end   = c_start + sliceIncrement;
        if (block == (n_blocks - 1) && leftOverSlices > 0) {
           c_end = c_start + leftOverSlices;                
        }
        hsize_t n_channels = (c_end - c_start);
        hsize_t block_size = n_channels * height * width;

        // Read the block into standardCube
        TIMER(timer.start("Read"););
        auto start_io = std::chrono::high_resolution_clock::now();
        readFitsData(inputFilePtr, c_start, s, block_size, standardCube, swapStokesFreqAxis);
        auto end_io = std::chrono::high_resolution_clock::now();
        total_io_ms += double(std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io).count());

        TIMER(timer.start("Histograms"););

        // 1. Parallel loop over channels to match the class's architectural design
#pragma omp parallel for default(none) shared(std::cout, progress, standardCube, height, width, statsXYZ, statsXY, c_start, c_end, cubeMin, cubeRange, channelProgressStride)
        for (hsize_t c = c_start; c < c_end; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
                            
            double chanMin = statsXY.minVals[c];
            double chanMax = statsXY.maxVals[c];
            double chanRange = chanMax - chanMin;
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);

            if(!chanHist){
               continue;
            }
            
            // 2. Use the channel's block index (c - c_start) as a unique offset 
            // to guarantee thread-safety without locks
            hsize_t buffer_offset = c - c_start; 
            hsize_t channel_offset = buffer_offset * height * width;
            
            for (hsize_t y = 0; y < height; y++) {
                auto y_width = channel_offset + y * width;                
                
                for (hsize_t x = 0; x < width; x++) {
                    auto pos = y_width + x;
                    auto& val = standardCube[pos];
                    if (std::isfinite(val)) {
                        statsXYZ.accumulatePartialHistogram(val, cubeMin, cubeRange, buffer_offset);
                    }
                }
            }
        } // end of parallel channels in block
        
        // 3. Consolidate OUTSIDE the channel loop! 
        // We now consolidate all partial histograms generated by the threads exactly once per block.
        statsXYZ.consolidateAndClearPartialHistogram(0);

    } // end of blocks loop

    auto end1 = std::chrono::high_resolution_clock::now();
    total_second_pass_processing_ms = double(std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count());

    long int totalBinCount = 0;
    for(size_t b = 0; b < numBins; ++b) {
       int64_t binCount = statsXYZ.getBinCount(0, b);
       totalBinCount += binCount;
    }
    printf("Total exact bin count = %ld\n", (long int)totalBinCount);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of 2nd pass took: " << duration.count() << " milliseconds " << (float(duration.count())/1000.00) << " seconds" << std::endl;

    return total_second_pass_processing_ms;
}

IOCostBreakdown SmartFastConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                                hsize_t numBins,
                                                const IOCostModel& readModel,
                                                const IOCostModel& writeModel) {
    IOCostBreakdown result;

    std::vector<hsize_t> swizzledDims = {stokes, width, height, depth};

    int sliceIncrement = n_io_blocks;
    int sliceIncrementCount = (int)(depth / sliceIncrement);
    int leftOverSlices = (int)(depth % sliceIncrement);
    int nBlocks = sliceIncrementCount + (leftOverSlices > 0 ? 1 : 0);

    // ---------- Pass 1: per-block read/write/rotate/mipmap ----------
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
        // NOTE: writeHdf5Data(standardDataSet, ...) is called TWICE per block
        // in the current code (lines ~169 and ~285), with identical arguments
        // and nothing modifying standardCube in between -- modeled here as
        // happening twice to match the code as it stands; worth checking
        // whether that duplicate is intentional.
        std::vector<hsize_t> standardDims = {stokes, depth, height, width};
        std::vector<hsize_t> chunkDims    = {1, 1, TILE_SIZE, TILE_SIZE};
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            auto perBlock = estimateHyperslabIO(standardDims, chunkDims,
                                                 {1, nChannels, height, width}, sizeof(float));
            acc.add(repeatEstimate(perBlock, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet block write (x2, see note)"));
    }

    // calculateChannelHistogram-equivalent, XY stats accumulation, rotation
    // (in-memory copy into rotatedCube), and Z-stats accumulation into
    // globalCountersZ: all pure in-memory compute, zero I/O.

    if (depth > 1) {
        // writeHdf5Data(swizzledDataSet, ...): unchunked, and only n_channels
        // (partial) of the innermost depth axis is selected each call -> no
        // axis can merge -> width*height separate transactions per block,
        // each just n_channels*sizeof(float) bytes. This is the phase most
        // likely to dominate this converter's I/O cost.
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            auto perBlock = estimateHyperslabIO(swizzledDims, {},
                                                 {1, width, height, nChannels}, sizeof(float));
            acc.add(repeatEstimate(perBlock, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: swizzledDataSet block write"));
    }

    {
        // mipMaps.write(s, c_start): once per BLOCK (not per channel). Each
        // level's write always spans its full height/width for n_channels at
        // once -> still 1 contiguous transaction per level per block, just
        // covering n_channels worth of data instead of 1.
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            hsize_t mipXY = 1, hLevel = height, wLevel = width;
            do {
                mipXY *= 2;
                hLevel = (hsize_t)std::ceil((double)height / mipXY);
                wLevel = (hsize_t)std::ceil((double)width  / mipXY);
                IOOpEstimate perLevel{ 1, nChannels * hLevel * wLevel * sizeof(double),
                                           nChannels * hLevel * wLevel * sizeof(double) };
                acc.add(repeatEstimate(perLevel, stokes), writeModel);
            } while (2 * wLevel > MIN_MIPMAP_SIZE || 2 * hLevel > MIN_MIPMAP_SIZE);
        }
        result.phases.push_back(acc.toPhase("1st pass: mipmap block writes (all levels)"));
    }

    // Post-block global consolidation (statsXYZ/statsZ from already-accumulated
    // in-memory counters): pure compute, zero I/O.

    // ---------- 2nd pass: conditional on bApproximateCubeHistogram ----------
    if (!bApproximateCubeHistogram && depth > 1) {
        // doSecondPass here genuinely re-reads block-by-block (unlike
        // SmartConverter's version, which ignores its block arguments).
        PhaseAccumulator acc;
        for (int block = 0; block < nBlocks; block++) {
            hsize_t nChannels = sliceIncrement;
            if (block == nBlocks - 1 && leftOverSlices > 0) nChannels = leftOverSlices;
            hsize_t bytesPerOp = nChannels * height * width * sizeof(float);
            acc.add(repeatEstimate(IOOpEstimate{1, bytesPerOp, bytesPerOp}, stokes), readModel);
        }
        result.phases.push_back(acc.toPhase("2nd pass: FITS block reread (doSecondPass)"));
    }
    // else (bApproximateCubeHistogram, or depth<=1): calcApproxCubeHistogram
    // rebins from statsXY's already-computed channel histograms -- zero I/O.

    if (depth > 1) {
        // statsZ.write(...): ONE full-height/width write per sub-dataset per
        // stokes -- fully contiguous despite the dataset being unchunked,
        // since the whole extent is written at once. No tiling bottleneck here.
        {
            PhaseAccumulator acc;
            hsize_t elemSizes[] = {4, 4, 8, 8, 8};
            for (auto es : elemSizes) {
                auto e = estimateHyperslabIO({stokes, height, width}, {}, {1, height, width}, es);
                acc.add(repeatEstimate(e, stokes), writeModel);
            }
            result.phases.push_back(acc.toPhase("statsZ write (whole array, once per stokes)"));
        }
        // statsXYZ.write(...): combined basic+histogram -- by this point the
        // cube histogram is already computed (either branch above), so no
        // split/defer is needed, unlike SmartConverter's version.
        {
            PhaseAccumulator acc;
            hsize_t elemSizes[] = {4, 4, 8, 8, 8};
            for (auto es : elemSizes) {
                auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
                acc.add(repeatEstimate(e, stokes), writeModel);
            }
            if (numBins > 0) {
                auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, 8);
                acc.add(repeatEstimate(h, stokes), writeModel);
            }
            result.phases.push_back(acc.toPhase("statsXYZ write (basic + histogram)"));
        }
    }

    // statsXY.write(...): combined basic+histogram, once per stokes, always
    // (not gated on depth > 1).
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
        result.phases.push_back(acc.toPhase("statsXY write (basic + histogram)"));
    }

    return result;
}