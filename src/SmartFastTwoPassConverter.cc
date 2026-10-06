/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "Converter.h"
#include <algorithm> // for std::min

SmartFastTwoPassConverter::SmartFastTwoPassConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips) 
 : SmartFastConverter(inputFileName, outputFileName, progress, zMips)
{}

// Memory model -- mirrors the allocations made in copyAndCalculate() and in
// SmartConverter::calculateRotatedDataAndCubeHistogram(), grouped by lifetime:
//
//   persistent  : allocated in pass 1 and kept until the converter is destroyed
//                 (statsXY, statsXYZ, mipmap buffers) -> present in BOTH passes
//   pass 1      : standardCube (n_io_blocks channels), freed before the rotation pass
//   rotation    : standardSlice + rotatedSlice (one full-depth tile) and tile-sized statsZ
//
//   peak = persistent + max(pass 1, rotation)
//
// NOTE: m.sizes lists every component for the report, but they do not all coexist,
//       so their sum is NOT the peak -- m.total is.
MemoryUsage SmartFastTwoPassConverter::calculateMemoryUsage() {
    MemoryUsage m;
    const hsize_t nBlock = (hsize_t)n_io_blocks; // channels per block (= sliceIncrement in copyAndCalculate)

    std::cout << "MEMORY_ESTIMATE: (SmartFastTwoPassConverter::calculateMemoryUsage) parameters: n_io_blocks = " << n_io_blocks << " , depth = " << depth << " , height = " << height << " , width = " << width << " , numBins = " << numBins << std::endl;

    // ---------------- persistent (both passes) ----------------
    // statsXY.createBuffers({depth}) -> partialHistMultiplier = 0
    m.sizes["XY stats"] = Stats::size({depth}, numBins);
    hsize_t persistent = m.sizes["XY stats"];

    if (depth > 1) {
        // statsXYZ.createBuffers({}, 0) -> one basic-stats entry + one cube histogram
        m.sizes["XYZ stats"] = Stats::size({}, numBins);
        persistent += m.sizes["XYZ stats"];
    }

    // mipMaps.createBuffers({n_io_blocks, height, width}). The smaller re-allocation for the last,
    // partial block frees the old buffers first (MipMap::createBuffers fix), so this is the maximum.
    // The buffers are only released in the destructor, i.e. they are still allocated during rotation.
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {nBlock, height, width}, zMips);
    persistent += m.sizes["Mipmaps"];

    // ---------------- pass 1 ----------------
    // standardCube = new float[height * width * sliceIncrement]
    m.sizes["Main dataset (pass 1 block)"] = nBlock * height * width * sizeof(float);
    hsize_t total_pass1 = m.sizes["Main dataset (pass 1 block)"];

    // ---------------- rotation pass ----------------
    hsize_t total_rotation = 0;
    if (depth > 1) {
        // standardSlice + rotatedSlice: one full-depth tile each, for ONE Stokes at a time.
        // Must match sliceSize in SmartConverter::calculateRotatedDataAndCubeHistogram().
        // Always TILE_SIZE x TILE_SIZE, even when the image (or an edge tile) is smaller.
        hsize_t sliceSize = product(trimAxes({1, depth, TILE_SIZE, TILE_SIZE}, N));
        m.sizes["Rotation tile slices"] = 2 * sliceSize * sizeof(float);

        // statsZ.createBuffers({TILE_SIZE, TILE_SIZE}) -> no histograms
        m.sizes["Z stats (tile)"] = Stats::size({TILE_SIZE, TILE_SIZE});

        total_rotation = m.sizes["Rotation tile slices"] + m.sizes["Z stats (tile)"];
    }

    m.total = persistent + std::max(total_pass1, total_rotation);

    std::cout << "MEMORY_ESTIMATE: persistent (XY/XYZ stats, mipmaps) = " << persistent * 1e-9 << " GB" << std::endl;
    std::cout << "MEMORY_ESTIMATE: pass 1 (block buffer)              = " << total_pass1 * 1e-9 << " GB" << std::endl;
    std::cout << "MEMORY_ESTIMATE: rotation pass (tile buffers)       = " << total_rotation * 1e-9 << " GB" << std::endl;
    std::cout << "MEMORY_ESTIMATE: peak usage = persistent + max(pass 1, rotation) = " << m.total * 1e-9 << " GB" << std::endl;

    m.note = " (peak = persistent + max(pass 1, rotation); components above do not all coexist)";
    return m;
}

// The only parameter that changes this converter's memory footprint is n_io_blocks, and that is already
// reduced by checkMemoryUsage() in main.cc (halving from depth). The inherited SmartConverter version
// tunes height_divider / allowed_mipmaps_threads, which this converter does not use -- so it would only
// iterate without effect (and permanently lower allowed_mipmaps_threads).
bool SmartFastTwoPassConverter::ReduceMemoryUsage( hsize_t memoryLimit, int max_iter /*=10*/ ) {
    UNUSED(max_iter);
    return calculateMemoryUsage().total <= memoryLimit;
}

void SmartFastTwoPassConverter::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    
    if(memoryLimitInMb <= 0) {
       std::cerr << "ERROR : memory limit not set for the SmartConverter and it is strictly required -> exiting SmartFastTwoPassConverter::copyAndCalculate function" << std::endl;
       return;
    }
    
    // NOTE: Z-mipmaps are not supported by this channel-blocked converter: MipMap::write() uses
    // c_start as the Z offset in every mipmap dataset, which is wrong for Z-mips (should be c_start/mipZ,
    // and blocks would also need to be aligned to mipZ). Also, with mipZ > 1 different channels in the
    // channel-parallel mipmap loop accumulate into the same mip cell (data race).
    if (zMips) {
       std::cerr << "WARNING : Z-mipmaps are not supported by SmartFastTwoPassConverter -> Z-mipmaps will be incorrect" << std::endl;
    }
    
    int sliceIncrement = n_io_blocks;
    int sliceIncrementCount = depth/sliceIncrement;
    int leftOverSlices = (depth % sliceIncrement);    
    int n_blocks = sliceIncrementCount;
    if (leftOverSlices > 0) {
       n_blocks++;
    }
    std::cout << "DEBUG : n_blocks = " << n_blocks << " vs.  sliceIncrementCount = " << sliceIncrementCount << " and leftOverSlices = " << leftOverSlices << std::endl;
              
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    
    // Allocate one block of channels at a time.
    // FIX 2: no rotatedCube here any more -- the rotated (swizzled) dataset is produced entirely by the
    //        tiled rotation pass (calculateRotatedDataAndCubeHistogram), which reads back standardDataSet.
    TIMER(timer.start("Allocate"););
    std::cout << "MEMORY : stokes:" << stokes << " depth: " << depth << " TILE_SIZE:" << TILE_SIZE << " N:" << N << std::endl;
    std::cout << "MEMORY (SmartFastTwoPassConverter::copyAndCalculate): allocating standardCube with size " << double(height * width * sliceIncrement*sizeof(float))/1e9 << " GB " << std::endl << std::flush;
    standardCube = new float[height * width * sliceIncrement];
    rotatedCube = nullptr;
    
    // Allocate one stokes of stats at a time
    std::cout << "DEBUG : before statsXY.createBuffers({" << depth << ")" << std::endl;
    statsXY.createBuffers({depth});
        
    if (depth > 1) {
        std::cout << "DEBUG : before statsXYZ.createBuffers({},0)" << std::endl;
        // Cube histogram is accumulated directly (no partial histograms) in the rotation pass
        statsXYZ.createBuffers({}, 0);

        // FIX 3: statsZ is NOT allocated here any more. Z statistics are computed tile-by-tile in the
        //        rotation pass, which allocates statsZ with {TILE_SIZE, TILE_SIZE}. Allocating {height, width}
        //        here was unused and leaked (Stats::createBuffers does not free previous buffers).
    }

    // FIX 4: track the channel-depth of the mipmap buffers, so that they can be re-created whenever the
    //        number of channels in the block changes (shrink for the last, partial block AND grow back
    //        for the first block of the next Stokes). Previously the buffers stayed at leftOverSlices size
    //        after Stokes 0 -> out-of-bounds accumulation for Stokes >= 1.
    hsize_t mipBufferChannels = sliceIncrement;
    printf("DEBUG : before mipMaps.createBuffers({%d,%llu,%llu})\n", sliceIncrement, height, width);    
    mipMaps.createBuffers({(hsize_t)sliceIncrement, height, width});

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    std::vector<double> savedCubeMin(stokes), savedCubeMax(stokes);
    
    // ===================================== 1st pass (per Stokes) =====================================
    // FITS -> standardDataSet, XY stats, channel histograms, mipmaps, XYZ basic stats (min/max/sum/...)
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);
        PROGRESS("\tMain loop\t");
        
        // Clear channel histogram accumulation buffers ONCE per Stokes
        statsXY.clearHistogramBuffers();
        
        double total_first_pass_processing_ms = 0.00;
        for (hsize_t block = 0; block < (hsize_t)n_blocks; block++ ) {
            hsize_t c_start = block*sliceIncrement;
            hsize_t c_end   = c_start + sliceIncrement;
            if (block == (hsize_t)(n_blocks-1) && leftOverSlices > 0 ) {
               c_end   = c_start + leftOverSlices; // if last block and there are left over slices                
            }
            hsize_t n_channels = (c_end-c_start);
            hsize_t block_size = n_channels*height*width;
            
            // FIX 4: re-create mipmap buffers whenever the block depth differs from the current buffers
            if (n_channels != mipBufferChannels) {
                mipMaps.createBuffers({n_channels, height, width});
                mipBufferChannels = n_channels;
            }
            
            std::cout << "DEBUG : processing block = " << block << " -> channel range " << c_start << " - " << c_end << std::endl;
            DEBUG(std::cout << " Reading main dataset..." << std::flush;);
            TIMER(timer.start("Read"););
            auto start_io = std::chrono::high_resolution_clock::now();
            readFitsData(inputFilePtr, c_start, s, block_size, standardCube, swapStokesFreqAxis);

            // Write the standard dataset
            DEBUG(std::cout << " Writing main dataset..." << std::flush;);
            TIMER(timer.start("Write"););
            std::vector<hsize_t> count = trimAxes({1, n_channels, height, width}, N);
            std::vector<hsize_t> memDims = {n_channels, height, width};
            std::vector<hsize_t> start = trimAxes({s, c_start, 0, 0}, N);
            writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = ms_d(end_io - start_io);
            auto block_io_ms = double(duration_io.count());
            std::cout << "I/O (readFitsData+writeHdf5Data) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;

            // XY stats (channel-parallel, each thread owns its channel index)
            auto start_processing = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (hsize_t i = c_start; i < c_end; i++) {
                PROGRESS_DECIMATED(i, channelProgressStride, "|");
                
                StatsCounter counterXY;
                counterXY.reset();
                
                for (hsize_t j = 0; j < height; j++) {
                    for (hsize_t k = 0; k < width; k++) {
                        auto sourceIndex = k + width * j + (height * width) * (i - c_start);
                        auto& val = standardCube[sourceIndex];
                        if (std::isfinite(val)) {
                            counterXY.accumulateFinite(val);
                        } else {
                            counterXY.accumulateNonFinite();
                        }
                    }
                }
                statsXY.copyStatsFromCounter(i, height * width, counterXY);
            }
            PROGRESS(std::endl);

            // Channel histograms (channel-parallel, each thread owns its channel's histogram)
            DEBUG(std::cout << " Channel Histograms..." << std::flush;);
            PROGRESS("\tChannel Histograms\t");
            TIMER(timer.start("Histograms"););
            #pragma omp parallel for
            for (hsize_t i = c_start; i < c_end; i++) {
                PROGRESS_DECIMATED(i, channelProgressStride, "|");
                
                double chanMin = statsXY.minVals[i];
                double chanMax = statsXY.maxVals[i];
                double chanRange = chanMax - chanMin;
                bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
                if (!chanHist) {
                    continue; 
                }

                for (hsize_t j = 0; j < width * height; j++) {
                    auto sourceIndex = (i - c_start) * width * height + j;
                    auto& val = standardCube[sourceIndex];
                    if (std::isfinite(val)) {
                        statsXY.accumulateHistogram(val, chanMin, chanRange, i);
                    }
                } 
            }                     
            auto end_processing = std::chrono::high_resolution_clock::now();
            auto duration_processing = ms_d(end_processing - start_processing);
            auto block_first_pass_processing_ms = double(duration_processing.count());

            // FIX 2: the per-block writeHdf5Data(swizzledDataSet, rotatedCube, ...) was removed here.
            //        rotatedCube was never filled in this converter, so it wrote uninitialised memory
            //        (later overwritten by the rotation pass, but at the cost of the most expensive write).

            // Mipmaps (channel-parallel; safe without Z-mips because each channel maps to its own z plane)
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
            }
            PROGRESS(std::endl);
            mipMaps.calculate();
            end_processing = std::chrono::high_resolution_clock::now();
            duration_processing = ms_d(end_processing - start_processing);
            block_first_pass_processing_ms += double(duration_processing.count());

            // Write the mipmaps for this block (Z offset = c_start)
            TIMER(timer.start("Write"););
            PROGRESS("\tWrite mipmaps" << std::endl);
            start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(s, c_start);        
            end_io = std::chrono::high_resolution_clock::now();
            duration_io = ms_d(end_io - start_io);
            block_io_ms += double(duration_io.count());
            std::cout << "I/O (mipMaps.write) for block : " << block << " took " << duration_io.count() << " milliseconds." << std::endl;
            
            // Clear the mipmaps before the next BLOCK
            TIMER(timer.start("Mipmaps"););
            mipMaps.resetBuffers();
            
            total_io_ms += block_io_ms;
            total_first_pass_processing_ms += block_first_pass_processing_ms;
            std::cout << "BENCHMARKING : block = " << block << " I/O took " << block_io_ms/1000.00 << " sec -> total I/O time = " << total_io_ms/1000.00 << " sec." << std::endl;
            std::cout << "BENCHMARKING : block = " << block << " pure processing took " << block_first_pass_processing_ms/1000.00 << " sec -> total pure processing took " << total_first_pass_processing_ms/1000.00 << " sec." << std::endl;
        } // end of loop over blocks
        
        // === POST-BLOCK: XYZ basic stats for this Stokes ===
        if (depth > 1) {
            auto start_processing = std::chrono::high_resolution_clock::now();            
            StatsCounter counterXYZ;
            for (hsize_t i = 0; i < depth; i++) {
                statsXY.accumulateStatsToCounter(counterXYZ, i);
            }
            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);
            
            // cube min/max are cached per Stokes for the rotation pass (it runs after ALL Stokes are done,
            // by which time the statsXYZ buffers hold only the last Stokes)
            savedCubeMin[s] = statsXYZ.minVals[0];
            savedCubeMax[s] = statsXYZ.maxVals[0];
            auto end_processing = std::chrono::high_resolution_clock::now();
            total_first_pass_processing_ms += double(ms_d(end_processing - start_processing).count());

            // FIX 1: write ONLY the basic XYZ stats here. The cube histogram is written by
            //        calculateRotatedDataAndCubeHistogram() once it has been filled in.
            auto start_io = std::chrono::high_resolution_clock::now();
            auto basicN = statsXYZ.basicDatasetDims.size();
            statsXYZ.writeBasic(statsXYZ.fullBasicBufferDims, trimAxes({1}, basicN), trimAxes({(hsize_t)s}, basicN));
            auto end_io = std::chrono::high_resolution_clock::now();
            total_io_ms += double(ms_d(end_io - start_io).count());
        }
        
        std::cout << "BENCHMARKING : total pure-processing time of 1st pass (Stokes " << s << "): " << total_first_pass_processing_ms << " milliseconds " << (float(total_first_pass_processing_ms)/1000.00) << " seconds" << std::endl;
        total_pureprocessing_ms += total_first_pass_processing_ms;
        
        // Write completed XY channel stats (basic + channel histograms)
        auto start_io = std::chrono::high_resolution_clock::now();
        statsXY.write({1, depth}, {(hsize_t)s, 0});
        auto end_io = std::chrono::high_resolution_clock::now();
        total_io_ms += double(ms_d(end_io - start_io).count());
    } // end of stokes
    
    // Free the 1st-pass block buffer BEFORE the rotation pass, so the two passes don't overlap in memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););
    delete[] standardCube;
    standardCube = nullptr;
    
    // ================================ 2nd pass: tiled rotation ================================
    // FIX 1: called ONCE, after all Stokes have been written to standardDataSet. The function loops over
    //        Stokes itself, reads standardDataSet tile-by-tile (full depth), writes the swizzled dataset
    //        and Z stats per tile, and computes + writes the exact cube (XYZ) histogram per Stokes.
    if (depth > 1) {
        auto start2 = std::chrono::high_resolution_clock::now();
        double total_rotation_pass_processing_ms = calculateRotatedDataAndCubeHistogram(total_io_ms, savedCubeMin, savedCubeMax);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::cout << "Execution of rotation pass (incl. Z stats and exact cube histogram) took: " << ms_d(end2 - start2).count() << " milliseconds." << std::endl;
        total_pureprocessing_ms += total_rotation_pass_processing_ms;
    }

    std::cout << "BENCHMARKING : total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;
    std::cout << "BENCHMARKING : total pure-processing time of 1st and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = ms_d(end - start);
    std::cout << "Execution of entire SmartFastTwoPassConverter::copyAndCalculate took: " << duration.count() << " milliseconds " << (float(duration.count())/1000.00) << " seconds" << std::endl;
    double unaccounted_for_ms = duration.count() - total_pureprocessing_ms - total_io_ms;
    std::cout << "Unaccounted for: " << unaccounted_for_ms/1000.00 << " seconds" << std::endl;
}

// I/O cost model for SmartFastTwoPassConverter -- mirrors copyAndCalculate() call by call:
//
//   Pass 1 (per Stokes, per block of n_io_blocks channels):
//       FITS block read -> standardDataSet block write -> mipmap writes (all XY levels)
//     then per Stokes: statsXY write (basic + channel histograms), statsXYZ writeBasic
//   Rotation pass (SmartConverter::calculateRotatedDataAndCubeHistogram, per Stokes, per 512x512 tile):
//       standardDataSet full-depth tile read -> swizzledDataSet tile write -> statsZ tile write
//     then per Stokes: statsXYZ writeHistogram (exact cube histogram)
//
// Dataset layouts (chunked vs contiguous) must match what Converter::convert() creates --
// keep the chunk-dims helpers below in sync with it.
IOCostBreakdown SmartFastTwoPassConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                                hsize_t numBins,
                                                const IOCostModel& readModel,
                                                const IOCostModel& writeModel) {
    IOCostBreakdown result;

    const hsize_t sliceIncrement = (hsize_t)n_io_blocks;
    const hsize_t leftOverSlices = depth % sliceIncrement;
    const hsize_t nBlocks = depth / sliceIncrement + (leftOverSlices > 0 ? 1 : 0);
    auto blockChannels = [&](hsize_t block) {
        return (block == nBlocks - 1 && leftOverSlices > 0) ? leftOverSlices : sliceIncrement;
    };

    // ---- on-disk layouts (as created in Converter::convert()) ----
    const std::vector<hsize_t> standardDims4 = {stokes, depth, height, width};
    const std::vector<hsize_t> swizzledDims4 = {stokes, width, height, depth};
    const std::vector<hsize_t> statsZDims    = {stokes, height, width};

    // standardDataSet: chunked {1,1,TILE,TILE} only if both image dims >= TILE_SIZE (useChunks)
    const std::vector<hsize_t> standardChunks =
        useChunks({height, width}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE} : std::vector<hsize_t>{};

    // swizzledDataSet: contiguous unless -C (rotatedDatasetChunking). This converter writes full-depth
    // tiles, so with -C it should get full-depth chunks like SMART-XY-PARALLEL (see Converter.cc).
    std::vector<hsize_t> swizzledChunks;
    if (Converter::rotatedDatasetChunking) {
        hsize_t chunkWidth  = std::min((hsize_t)TILE_SIZE, width);
        hsize_t chunkHeight = std::min((hsize_t)TILE_SIZE, height);
        const hsize_t MAX_CHUNK_BYTES = 2048ULL * 1024ULL * 1024ULL;
        while (chunkWidth * chunkHeight * depth * sizeof(float) > MAX_CHUNK_BYTES && (chunkWidth > 1 || chunkHeight > 1)) {
            if (chunkWidth >= chunkHeight && chunkWidth > 1) chunkWidth = std::max((hsize_t)1, chunkWidth / 2);
            else if (chunkHeight > 1)                         chunkHeight = std::max((hsize_t)1, chunkHeight / 2);
        }
        swizzledChunks = {1, chunkWidth, chunkHeight, depth};
    }

    // statsZ: chunked {1, min(TILE,H), min(TILE,W)}
    const std::vector<hsize_t> statsZChunks = {1, std::min((hsize_t)TILE_SIZE, height), std::min((hsize_t)TILE_SIZE, width)};

    // On-disk element sizes of the 5 basic-stats datasets: MIN, MAX, SUM, SUM_SQ are created as
    // float (4 bytes, the double buffers are converted on write), NAN_COUNT as int64 (8 bytes).
    const hsize_t basicElemSizes[] = {4, 4, 4, 4, 8};

    // ======================= Pass 1 =======================
    {
        // readFitsData: one contiguous read of n_channels full images per block
        PhaseAccumulator acc;
        for (hsize_t block = 0; block < nBlocks; block++) {
            hsize_t bytes = blockChannels(block) * height * width * sizeof(float);
            acc.add(repeatEstimate(IOOpEstimate{1, bytes, bytes}, stokes), readModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: FITS block read (readFitsData)"));
    }
    {
        // writeHdf5Data(standardDataSet, ...) once per block
        PhaseAccumulator acc;
        for (hsize_t block = 0; block < nBlocks; block++) {
            auto e = estimateHyperslabIO(standardDims4, standardChunks,
                                         {1, blockChannels(block), height, width}, sizeof(float));
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet block write"));
    }
    {
        // mipMaps.write(s, c_start): one write per XY level per block. Levels mirror the MipMaps
        // constructor (no Z-mips -- not supported by this converter). Datasets are float on disk,
        // chunked with tileDims only if the level is >= TILE_SIZE in both dims (useChunks).
        PhaseAccumulator acc;
        for (hsize_t block = 0; block < nBlocks; block++) {
            hsize_t nCh = blockChannels(block);
            hsize_t hLevel = height, wLevel = width;
            int mipXY = 1;
            do {
                if (mipXY > 1) {
                    std::vector<hsize_t> levelDims = {stokes, depth, hLevel, wLevel};
                    std::vector<hsize_t> levelChunks =
                        useChunks({hLevel, wLevel}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE} : std::vector<hsize_t>{};
                    auto e = estimateHyperslabIO(levelDims, levelChunks, {1, nCh, hLevel, wLevel}, sizeof(float));
                    acc.add(repeatEstimate(e, stokes), writeModel);
                }
                mipXY *= 2;
                hLevel = (hsize_t)std::ceil((float)hLevel / 2);
                wLevel = (hsize_t)std::ceil((float)wLevel / 2);
            } while (2 * wLevel > MIN_MIPMAP_SIZE || 2 * hLevel > MIN_MIPMAP_SIZE);
        }
        result.phases.push_back(acc.toPhase("1st pass: mipmap block writes (all XY levels)"));
    }
    {
        // statsXY.write({1, depth}, {s, 0}): 5 basic datasets + channel histograms, once per Stokes
        PhaseAccumulator acc;
        for (auto es : basicElemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, sizeof(int64_t));
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: statsXY write (basic + channel histograms)"));
    }
    if (depth > 1) {
        // statsXYZ.writeBasic(...): 5 single values per Stokes (histogram is written in the rotation pass)
        PhaseAccumulator acc;
        for (auto es : basicElemSizes) {
            auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: statsXYZ writeBasic"));
    }

    // ======================= Rotation pass (depth > 1 only) =======================
    if (depth > 1) {
        PhaseAccumulator readAcc, writeAcc, statsZAcc;

        // same tile grid as calculateRotatedDataAndCubeHistogram, so edge tiles get their own size
        for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
            hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
            for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                hsize_t ySize = std::min(TILE_SIZE, height - yOffset);

                // readHdf5Data(standardDataSet, ...) of {1, depth, ySize, xSize}
                auto r = estimateHyperslabIO(standardDims4, standardChunks, {1, depth, ySize, xSize}, sizeof(float));
                readAcc.add(repeatEstimate(r, stokes), readModel);

                // writeHdf5Data(swizzledDataSet, ...) of {1, xSize, ySize, depth}
                auto w = estimateHyperslabIO(swizzledDims4, swizzledChunks, {1, xSize, ySize, depth}, sizeof(float));
                writeAcc.add(repeatEstimate(w, stokes), writeModel);

                // statsZ.write(...) of {1, ySize, xSize} into each of the 5 basic datasets
                for (auto es : basicElemSizes) {
                    auto z = estimateHyperslabIO(statsZDims, statsZChunks, {1, ySize, xSize}, es);
                    statsZAcc.add(repeatEstimate(z, stokes), writeModel);
                }
            }
        }
        result.phases.push_back(readAcc.toPhase("Rotation pass: standardDataSet tile read"));
        result.phases.push_back(writeAcc.toPhase(std::string("Rotation pass: swizzledDataSet tile write") +
                                                 (swizzledChunks.empty() ? " (contiguous)" : " (chunked)")));
        result.phases.push_back(statsZAcc.toPhase("Rotation pass: statsZ tile write"));

        if (numBins > 0) {
            // statsXYZ.writeHistogram(...): the exact cube histogram, once per Stokes
            PhaseAccumulator acc;
            auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, sizeof(int64_t));
            acc.add(repeatEstimate(h, stokes), writeModel);
            result.phases.push_back(acc.toPhase("Rotation pass: statsXYZ writeHistogram"));
        }
    }

    return result;
}
