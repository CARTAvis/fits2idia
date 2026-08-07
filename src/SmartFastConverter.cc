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
       std::cout << "MEMORY_ESTIMATE: Z stats  = " << m.sizes["Z stats"]* 1e-9 << " GB" << std::endl;
       
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
    
    // calculate memory limits in different units, slice here is image in a single freq. channel:
    double memoryLimitInBytes = double(memoryLimitInMb) * 1024.00 * 1024.00;
    double double_memoryLimitInBytes = double(memoryLimitInMb) * 1024.0 * 1024.0;
    // FIX: Multiply by 4.0 to account for standardCube (1x), rotatedCube (1x), 
    // mipMaps (~1.3x), and the persistent globalCountersZ overhead.
    double sliceSizeInPixels = height * width * 4;
    double memoryLimitInPixels = memoryLimitInBytes / sizeof(float);
    std::cout << "DEBUG memoryLimitInMb = " << memoryLimitInMb << " -> memoryLimitInBytes = " << memoryLimitInBytes << " -> memoryLimitInPixels = " << memoryLimitInPixels << std::endl;
    std::cout << "DEBUG double_memoryLimitInBytes = " << double_memoryLimitInBytes << std::endl;
    double memoryLimitInSlices = std::ceil(memoryLimitInPixels / sliceSizeInPixels);
    // first use MAX of memoryLimitInSlices and 1 , and then make sure we are not trying to read more channels than exist -> min(depth, MAX)
    int sliceIncrement = std::max(memoryLimitInSlices, (double)1.00); // first make sure we use at least 1 slice
    if (n_io_blocks > 1) {
       sliceIncrement = n_io_blocks;
    }
    sliceIncrement = std::min( int(depth), sliceIncrement ); // then make sure we do not read more channels than there are in FITS file 
    std::cout << "DEBUG : final sliceIncrement = " << sliceIncrement << " (n_io_blocks = " << n_io_blocks << ")" << std::endl;
    int sliceIncrementCount = depth/sliceIncrement;                   // number of portions to be read 
    int leftOverSlices = (depth % sliceIncrement);    
    std::cout << "SIZEOF(float) = " << sizeof(float) << ", Image size:" << height << " x " << width << std::endl;
    std::cout << "MEMORY limits " << memoryLimitInMb << " MB = " << memoryLimitInPixels << " pixels = " << memoryLimitInSlices 
              << " slices -> sliceIncrement = " << sliceIncrement << " sliceIncrementCount = " << depth << "/" << sliceIncrement << " = " << sliceIncrementCount 
              << " -> leftover slices = " << leftOverSlices 
              << std::endl;
    // 
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
            total_io_ms += double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;

                    
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
            

            DEBUG(std::cout << " Writing main and rotated datasets... " << std::flush;);
            PROGRESS("\tWrite data" << std::endl);
            TIMER(timer.start("Write"););
                 
            start_io = std::chrono::high_resolution_clock::now();                        
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
        
            if (depth > 1) {
                // This all technically worked if we reused the standard filespace and memspace
                // But it's probably not a good idea to rely on two incorrect values cancelling each other out
                // Use n_channels instead of depth            
                std::vector<hsize_t> swizzledCount = trimAxes({1, width, height, n_channels}, N);
                std::vector<hsize_t> swizzledMemDims = {width, height, n_channels};            
                // Standard format is {Stokes, Depth, Height, Width} -> Start is {s, c_start, 0, 0}
                // Swizzled format is {Stokes, Width, Height, Depth} -> Start is {s, 0, 0, c_start}
                std::vector<hsize_t> swizzledStart = trimAxes({currentStokes, 0, 0, c_start}, N);
                
                writeHdf5Data(swizzledDataSet, rotatedCube, swizzledMemDims, swizzledCount, swizzledStart);
            }
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());

            // Fourth loop handles mipmaps        
            // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
            DEBUG(std::cout << " Mipmaps..." << std::endl;);
            PROGRESS("\tMipmaps\t\t");
            TIMER(timer.start("Mipmaps"););

                
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
            
            TIMER(timer.start("Write"););
            PROGRESS("\tWrite stats & mipmaps" << std::endl);
            
            // Write the mipmaps
            // FIX: Pass c_start so the HDF5 writer knows the correct Z-offset
            start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(currentStokes, c_start);        
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
            
            // Write the statistics                
            // Clear the mipmaps before the next BLOCK (not Stokes)
            TIMER(timer.start("Mipmaps"););
            mipMaps.resetBuffers();
        } // end of loop over blocks
        
        // === POST-BLOCK GLOBAL CONSOLIDATION ===
        if (depth > 1) {
            std::cout << "Finalizing Global XYZ and Z statistics..." << std::endl;
            
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
            
            // 4. Write Z stats and the fully completed Global XYZ object
            auto start_io = std::chrono::high_resolution_clock::now();
            statsZ.write({height, width}, {1, height, width}, {currentStokes, 0, 0});
            statsXYZ.write({1}, {currentStokes});
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io);
            total_io_ms += double(duration_io.count());
        }
        
        // 5. Write completed XY channel stats
        statsXY.write({1, depth}, {currentStokes, 0});

    } // end of stokes
    
    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    
    delete[] standardCube;
    if (depth > 1) {               // ADD THIS
        delete[] rotatedCube;      // ADD THIS
    }

    std::cout << "Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;
    std::cout << "BENCHMARKING : total pure-processing time of 1st, 2nd and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution of entire SmartConverter::copyAndCalculate took: " << duration.count() << " milliseconds " << (float(duration.count())/1000.00) << " seconds" << std::endl;
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

