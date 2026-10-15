/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// FastConverterLimitedMemory
// --------------------------
// Same algorithm as FastConverter (whole Stokes cube read into memory, a single FITS read per Stokes,
// channel-parallel XY stats / histograms / mipmaps), but the rotated (swizzled) dataset and the Z
// statistics are NOT materialised for the whole cube. Instead they are produced in spatial strips:
//
//     strip = columns [x0, x0 + nx) x all rows x all channels
//
// The swizzled dataset has the on-disk layout {stokes, width, height, depth}, i.e. x is the SLOWEST
// varying spatial axis. A strip of full-height, full-depth columns is therefore one CONTIGUOUS region
// of the swizzled dataset, so every strip is written with one large sequential write (no
// fragmentation, unlike SmartFastConverter's {1, W, H, n_channels} writes or the tile writes of the
// two-pass converter into a contiguous dataset).
//
// Mipmaps are computed from the in-memory cube in blocks of mipChannelBlock channels (written with
// the channel offset, like SmartFastConverter), and their buffers are freed after each Stokes, so
// the rotation strip and the mipmap buffers never coexist. This matters: the mipmap buffers are
// double + int per mip cell, i.e. ~12 B per cell x ~1/3 of the cube's elements ~= the cube size.
//
// Peak memory:  cube + XY/XYZ stats + statsZ(strip) + max(rotation strip, mipmap block)
// FastConverter:  cube + rotated cube + full mipmaps + full statsZ (all allocated at the same time)
//
// With no memory limit (or enough memory) stripWidth == width and mipChannelBlock == depth, and the
// output is identical to FastConverter's.

#include "Converter.h"
#include <algorithm>

FastConverterLimitedMemory::FastConverterLimitedMemory(std::string inputFileName, std::string outputFileName, bool progress, bool zMips)
    : FastConverter(inputFileName, outputFileName, progress, zMips), memoryLimitInMb(0), stripWidth(0), mipChannelBlock(0) {
    // default: behave exactly like FastConverter (single strip, all channels' mipmaps at once)
    stripWidth = width;
    mipChannelBlock = depth;
}

void FastConverterLimitedMemory::setMemoryLimit(int _memoryLimitInMb) {
    memoryLimitInMb = _memoryLimitInMb;
    if (memoryLimitInMb > 0) {
        hsize_t memoryLimit = (hsize_t)(memoryLimitInMb * 1e6); // same MB convention as main.cc
        if (!fitToMemory(memoryLimit)) {
            // does not fit even with the smallest strip / mipmap block: keep the minimum so that
            // calculateMemoryUsage() reports the true minimum and checkMemoryUsage() rejects it
            stripWidth = 1;
            mipChannelBlock = minMipChannelBlock();
        }
    } else {
        stripWidth = width;
        mipChannelBlock = depth;
    }
    std::cout << "INFO (FastConverterLimitedMemory) : memory limit = " << memoryLimitInMb << " MB -> strip width = "
              << stripWidth << " columns (" << ((width + stripWidth - 1) / stripWidth) << " strips per Stokes), "
              << "mipmap block = " << mipChannelBlock << " channels" << std::endl;
}

// Z-mipmaps need all channels at once: MipMap::write() uses the channel offset as the Z offset in every
// mipmap dataset, which is only correct for XY-only mipmaps.
hsize_t FastConverterLimitedMemory::minMipChannelBlock() {
    return zMips ? depth : 1;
}

// Memory alive during the whole conversion, independent of strip width and mipmap block:
// whole Stokes cube + XY and XYZ stats (Stats buffers cannot be freed, see Stats::createBuffers).
hsize_t FastConverterLimitedMemory::fixedMemory() {
    hsize_t fixed = depth * height * width * sizeof(float);          // standardCube
    fixed += Stats::size({depth}, numBins);                          // statsXY
    if (depth > 1) {
        fixed += Stats::size({}, numBins, depth);                    // statsXYZ incl. partial histograms
    }
    return fixed;
}

// Z stats for one strip: allocated once, kept for the whole conversion (Stats has no free()).
hsize_t FastConverterLimitedMemory::statsZMemory(hsize_t sw) {
    return (depth > 1) ? Stats::size({height, sw}) : 0;
}

// Rotated strip buffer: allocated for the rotation phase of each Stokes, freed before the mipmaps.
hsize_t FastConverterLimitedMemory::stripMemory(hsize_t sw) {
    return (depth > 1) ? sw * height * depth * sizeof(float) : 0;
}

// Mipmap buffers for a block of cb channels: allocated for the mipmap phase, freed afterwards.
hsize_t FastConverterLimitedMemory::mipMemory(hsize_t cb) {
    return MipMaps::size(standardDims, {cb, height, width}, zMips);
}

hsize_t FastConverterLimitedMemory::peakMemory(hsize_t sw, hsize_t cb) {
    return fixedMemory() + statsZMemory(sw) + std::max(stripMemory(sw), mipMemory(cb));
}

// Chooses stripWidth and mipChannelBlock for the given limit. Strip width is maximised first (subject to
// the smallest legal mipmap block still fitting), then the mipmap block is maximised for that strip width.
// Strip widths are rounded down to multiples of TILE_SIZE when possible, so strips stay aligned with the
// statsZ chunks. Returns false if nothing fits.
bool FastConverterLimitedMemory::fitToMemory(hsize_t memoryLimit) {
    const hsize_t cbMin = minMipChannelBlock();

    if (peakMemory(width, depth) <= memoryLimit) {
        stripWidth = width;
        mipChannelBlock = depth;
        return true;
    }

    hsize_t sw = width;
    if (depth > 1) {
        if (peakMemory(1, cbMin) > memoryLimit) {
            return false;
        }
        // largest sw in [1, width] with peakMemory(sw, cbMin) <= limit (monotonic in sw)
        hsize_t lo = 1, hi = width;
        while (lo < hi) {
            hsize_t mid = lo + (hi - lo + 1) / 2;
            if (peakMemory(mid, cbMin) <= memoryLimit) lo = mid; else hi = mid - 1;
        }
        if (lo >= (hsize_t)TILE_SIZE) {
            lo = (lo / TILE_SIZE) * TILE_SIZE;
        }
        sw = lo;
    } else if (peakMemory(width, cbMin) > memoryLimit) {
        return false;
    }

    // largest cb in [cbMin, depth] with peakMemory(sw, cb) <= limit (monotonic in cb)
    hsize_t lo = cbMin, hi = depth;
    while (lo < hi) {
        hsize_t mid = lo + (hi - lo + 1) / 2;
        if (peakMemory(sw, mid) <= memoryLimit) lo = mid; else hi = mid - 1;
    }

    stripWidth = sw;
    mipChannelBlock = lo;
    return true;
}

bool FastConverterLimitedMemory::ReduceMemoryUsage(hsize_t memoryLimit, int max_iter /*=10*/) {
    UNUSED(max_iter);
    if (!fitToMemory(memoryLimit)) {
        std::cout << "WARNING (FastConverterLimitedMemory) : even the minimum configuration ("
                  << peakMemory(1, minMipChannelBlock()) * 1e-9 << " GB) does not fit in " << memoryLimit * 1e-9
                  << " GB -> use SmartFastTwoPassConverter / SlowConverter instead" << std::endl;
        return false;
    }
    return true;
}

MemoryUsage FastConverterLimitedMemory::calculateMemoryUsage() {
    MemoryUsage m;

    m.sizes["Main dataset"] = depth * height * width * sizeof(float);
    m.sizes["XY stats"] = Stats::size({depth}, numBins);
    m.sizes["Mipmaps (block of " + std::to_string(mipChannelBlock) + " channels)"] = mipMemory(mipChannelBlock);

    if (depth > 1) {
        m.sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        m.sizes["Rotation (strip)"] = stripMemory(stripWidth);
        m.sizes["Z stats (strip)"] = statsZMemory(stripWidth);
    }

    // rotation strip and mipmap buffers are never allocated at the same time
    m.total = peakMemory(stripWidth, mipChannelBlock);

    hsize_t nStrips = (width + stripWidth - 1) / stripWidth;
    hsize_t nMipBlocks = (depth + mipChannelBlock - 1) / mipChannelBlock;
    m.note = " (rotation in " + std::to_string(nStrips) + " strip(s) of " + std::to_string(stripWidth) +
             " columns, mipmaps in " + std::to_string(nMipBlocks) + " block(s); rotation strip and mipmaps are not allocated at the same time)";

    std::cout << "MEMORY_ESTIMATE: (FastConverterLimitedMemory) stripWidth = " << stripWidth
              << " , mipChannelBlock = " << mipChannelBlock
              << " , fixed = " << fixedMemory() * 1e-9 << " GB , statsZ(strip) = " << statsZMemory(stripWidth) * 1e-9
              << " GB , max(strip " << stripMemory(stripWidth) * 1e-9 << " , mipmaps " << mipMemory(mipChannelBlock) * 1e-9
              << ") GB , peak = " << m.total * 1e-9 << " GB" << std::endl;

    return m;
}

void FastConverterLimitedMemory::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));

    if (stripWidth == 0 || stripWidth > width) {
        stripWidth = width;
    }
    const hsize_t nStrips = (width + stripWidth - 1) / stripWidth;
    const hsize_t planeSize = height * width;
    const hsize_t cubeSize = depth * planeSize;

    std::cout << "INFO (FastConverterLimitedMemory) : strip width = " << stripWidth << " columns, "
              << nStrips << " strip(s) per Stokes" << std::endl;

    TIMER(timer.start("Allocate"););

    standardCube = new float[cubeSize];
    rotatedCube = nullptr;

    statsXY.createBuffers({depth});

    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
        // Z stats only for ONE strip. Allocated once (Stats::createBuffers does not free previous
        // buffers), the last, narrower strip just uses a prefix of these buffers.
        statsZ.createBuffers({height, stripWidth});

    }

    // rotated strip buffer and mipmap buffers are allocated per Stokes, in separate phases (see below)
    if (mipChannelBlock == 0 || mipChannelBlock > depth) {
        mipChannelBlock = depth;
    }
    if (zMips && mipChannelBlock != depth) {
        std::cerr << "WARNING : Z-mipmaps require all channels in one mipmap block -> using " << depth << std::endl;
        mipChannelBlock = depth;
    }

    // Largest XY mip factor: bands of this many rows map to disjoint mip cells at EVERY level, so mipmap
    // accumulation can be parallelised over row bands without races (and is not limited to depth threads).
    hsize_t maxMipXY = 1;
    for (auto& mipMap : mipMaps.mipMaps) {
        maxMipXY = std::max(maxMipXY, (hsize_t)mipMap.mipXY);
    }
    const hsize_t nBands = (height + maxMipXY - 1) / maxMipXY;

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        DEBUG(std::cout << "Processing Stokes " << currentStokes << "..." << std::endl;);
        PROGRESS("Stokes " << currentStokes << ":" << std::endl);

        // ------------------------------------------------------------------ read whole Stokes cube
        TIMER(timer.start("Read"););
        auto start_io = std::chrono::high_resolution_clock::now();
        readFitsData(inputFilePtr, 0, currentStokes, cubeSize, standardCube, swapStokesFreqAxis);
        auto end_io = std::chrono::high_resolution_clock::now();
        auto duration_io = ms_d(end_io - start_io);
        total_io_ms += double(duration_io.count());
        std::cout << "I/O (readFitsData) for Stokes : " << currentStokes << " took " << duration_io.count() << " milliseconds." << std::endl;

        // ------------------------------------------------------------------ 1st loop: XY stats
        // (identical to FastConverter, minus the rotation which is now done strip-wise below)
        PROGRESS("\tXY statistics\t");
        TIMER(timer.start("XY statistics"););
        auto start1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");
            StatsCounter counterXY;

            std::function<void(float)> accumulate;
            auto lazy_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazy(val);
            };
            auto first_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazyFirst(val);
                accumulate = lazy_accumulate;
            };
            accumulate = first_accumulate;

            const float* channel = standardCube + i * planeSize;
            for (hsize_t p = 0; p < planeSize; p++) {
                const float val = channel[p];
                if (std::isfinite(val)) {
                    accumulate(val);
                } else {
                    counterXY.accumulateNonFinite();
                }
            }

            statsXY.copyStatsFromCounter(i, planeSize, counterXY);
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        double duration1_ms = ms_d(end1 - start1).count();
        std::cout << "BENCHMARKING : total pure-processing time of 1st pass (statsXY): " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += duration1_ms;
        PROGRESS(std::endl);

        // ------------------------------------------------------------------ XYZ basic stats
        if (depth > 1) {
            TIMER(timer.start("XYZ statistics"););
            start1 = std::chrono::high_resolution_clock::now();
            StatsCounter counterXYZ;
            for (hsize_t i = 0; i < depth; i++) {
                statsXY.accumulateStatsToCounter(counterXYZ, i);
            }
            statsXYZ.copyStatsFromCounter(0, cubeSize, counterXYZ);
            end1 = std::chrono::high_resolution_clock::now();
            total_pureprocessing_ms += ms_d(end1 - start1).count();
        }

        // ------------------------------------------------------------------ histograms (unchanged)
        PROGRESS("\tHistograms\t");
        TIMER(timer.start("Histograms"););

        double cubeMin = 0.0, cubeMax = 0.0, cubeRange = 0.0;
        bool cubeHist(false);
        if (depth > 1) {
            cubeMin = statsXYZ.minVals[0];
            cubeMax = statsXYZ.maxVals[0];
            cubeRange = cubeMax - cubeMin;
            cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
        }

        start1 = std::chrono::high_resolution_clock::now();
        statsXY.clearHistogramBuffers();
        statsXYZ.clearHistogramBuffers();

#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");

            double chanMin = statsXY.minVals[i];
            double chanMax = statsXY.maxVals[i];
            double chanRange = chanMax - chanMin;
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);

            if (!chanHist && !cubeHist) {
                continue;
            }

            const float* channel = standardCube + i * planeSize;
            for (hsize_t p = 0; p < planeSize; p++) {
                const float val = channel[p];
                if (std::isfinite(val)) {
                    if (chanHist) {
                        statsXY.accumulateHistogram(val, chanMin, chanRange, i);
                    }
                    if (cubeHist) {
                        statsXYZ.accumulatePartialHistogram(val, cubeMin, cubeRange, i);
                    }
                }
            }
        }

        if (depth > 1) {
            statsXYZ.consolidatePartialHistogram();
        }
        end1 = std::chrono::high_resolution_clock::now();
        duration1_ms = ms_d(end1 - start1).count();
        std::cout << "BENCHMARKING : total pure-processing time of histograms pass: " << duration1_ms << " milliseconds " << duration1_ms/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += duration1_ms;
        PROGRESS(std::endl);

        // ------------------------------------------------------------------ write standard dataset
        PROGRESS("\tWrite main data" << std::endl);
        TIMER(timer.start("Write"););
        start_io = std::chrono::high_resolution_clock::now();
        {
            std::vector<hsize_t> memDims = {depth, height, width};
            std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
            std::vector<hsize_t> startPos = trimAxes({currentStokes, 0, 0, 0}, N);
            writeHdf5Data(standardDataSet, standardCube, memDims, count, startPos);
        }
        end_io = std::chrono::high_resolution_clock::now();
        total_io_ms += ms_d(end_io - start_io).count();

        // ------------------------------------------------------------------ strip-wise rotation + Z stats
        if (depth > 1) {
            PROGRESS("\tRotation + Z stats (" << nStrips << " strips)\t");
            double strip_proc_ms = 0.0, strip_io_ms = 0.0;

            TIMER(timer.start("Allocate"););
            std::cout << "MEMORY (FastConverterLimitedMemory): allocating rotated strip buffer with size "
                      << double(stripWidth * height * depth * sizeof(float)) / 1e9 << " GB" << std::endl;
            rotatedCube = new float[stripWidth * height * depth];

            for (hsize_t x0 = 0; x0 < width; x0 += stripWidth) {
                const hsize_t nx = std::min(stripWidth, width - x0);
                PROGRESS("|");

                // --- rotate strip: rotatedCube[kl][j][i] = standardCube[i][j][x0 + kl]
                // Cache-blocked transpose: destination is written contiguously along depth, source is
                // read in short contiguous row segments. Parallelised spatially, so it scales even when
                // depth is small (e.g. 10 channels), unlike the channel-parallel loop of FastConverter.
                TIMER(timer.start("Rotation"););
                auto sp = std::chrono::high_resolution_clock::now();

                // Memory layouts (floats):
                //   source, whole cube : standardCube[i][j][x]  -> index = x + W*j + P*i      (P = W*H, x fastest)
                //   destination, strip : rotatedCube[kl][j][i]  -> index = kl*H*D + j*D + i  (i fastest)
                //
                // A cache line is 64 B = 16 floats. Reading one float from memory brings in 16 neighbouring x values.
                // Goal: load every source line ONCE and use all 16 floats while it is still in cache.
                //       even though the first of 16 numbers is a miss the next 15 are hits
                const hsize_t BLK = 32;                 // tile edge: 32 rows x 32 columns x up to 32 channels
                const hsize_t strideK = height * depth; // distance (floats) between consecutive columns kl in dst

                // ---- Loop 1+2: split the strip (H rows x nx columns) into 32x32 tiles in (y, x).
                //      Each (jb, kb) tile is an independent unit of work given to one thread.
                //      Tiles write disjoint parts of rotatedCube -> no races, no shared cache lines.
                #pragma omp parallel for collapse(2) schedule(static)
                for (hsize_t jb = 0; jb < height; jb += BLK) {          // first row of the tile
                    for (hsize_t kb = 0; kb < nx; kb += BLK) {           // first column of the tile (local x)
                        const hsize_t jEnd = std::min(jb + BLK, height); // tile may be cut at the image edge
                        const hsize_t kEnd = std::min(kb + BLK, nx);     // tile may be cut at the strip edge

                        // ---- Loop 3: split channels into groups of 32 so the tile's cached working set stays
                        //      small even for cubes with thousands of channels. For depth = 8 this runs once.
                        for (hsize_t ib = 0; ib < depth; ib += BLK) {
                            const hsize_t iEnd = std::min(ib + BLK, depth);

                            // ---- Loop 4: columns of the tile, ONE AT A TIME. This is the reuse loop:
                            //      column kl+1 needs x+1, which sits in the SAME source cache lines that
                            //      column kl just loaded (16 consecutive x per line).
                            for (hsize_t kl = kb; kl < kEnd; kl++) {

                                // ---- Loop 5: rows of the tile.
                                for (hsize_t j = jb; j < jEnd; j++) {
                                    // dst: start of the spectrum of pixel (x0+kl, j) -- D contiguous floats.
                                    //      For fixed kl, consecutive j are also contiguous (j*D), so the
                                    //      whole j-loop writes one contiguous run of (jEnd-jb)*D floats.
                                    float* dst = rotatedCube + kl * strideK + j * depth;
                                    // src: pixel (x0+kl, j) in channel 0 of the source cube.
                                    const float* src = standardCube + (x0 + kl) + width * j;

                                    // ---- Loop 6: channels. Writes are contiguous (dst[i], dst[i+1], ...).
                                    //      Reads jump by a whole plane P per step: each read lands in a
                                    //      DIFFERENT cache line (different channel plane). The first time
                                    //      (kl = multiple of 16) these are misses; for the next 15 values
                                    //      of kl the same lines are hits.
                                    for (hsize_t i = ib; i < iEnd; i++) {
                                        dst[i] = src[planeSize * i];
                                    }
                                }
                            }
                        }
                    }
                }
                

                // --- Z stats for the strip, read from the rotated buffer (spectra are contiguous there).
                // Buffer index is dense for THIS strip (kl + j * nx), so the last, narrower strip uses
                // a contiguous prefix of the statsZ buffers and is written with bufferDims {height, nx}.
                TIMER(timer.start("Z statistics"););
#pragma omp parallel for collapse(2) schedule(static)
                for (hsize_t j = 0; j < height; j++) {
                    for (hsize_t kl = 0; kl < nx; kl++) {
                        StatsCounter counterZ;
                        const float* spectrum = rotatedCube + kl * strideK + j * depth;
                        for (hsize_t i = 0; i < depth; i++) {
                            const float val = spectrum[i];
                            if (std::isfinite(val)) {
                                // Not lazy; too much risk of encountering an ascending / descending sequence.
                                counterZ.accumulateFinite(val);
                            } else {
                                counterZ.accumulateNonFinite();
                            }
                        }
                        statsZ.copyStatsFromCounter(kl + j * nx, depth, counterZ);
                    }
                }
                auto ep = std::chrono::high_resolution_clock::now();
                strip_proc_ms += ms_d(ep - sp).count();

                // --- write the strip: {1, nx, height, depth} at {s, x0, 0, 0} -> one contiguous region
                //     of the (contiguous) swizzled dataset
                TIMER(timer.start("Write"););
                auto si = std::chrono::high_resolution_clock::now();
                {
                    std::vector<hsize_t> swizzledMemDims = {nx, height, depth};
                    std::vector<hsize_t> swizzledCount = trimAxes({1, nx, height, depth}, N);
                    std::vector<hsize_t> swizzledStart = trimAxes({currentStokes, x0, 0, 0}, N);
                    writeHdf5Data(swizzledDataSet, rotatedCube, swizzledMemDims, swizzledCount, swizzledStart);

                    // statsZ: write(bufferDims, count, start) -- same call convention as SmartFastConverter
                    statsZ.write({height, nx}, {1, height, nx}, {currentStokes, 0, x0});
                }
                auto ei = std::chrono::high_resolution_clock::now();
                strip_io_ms += ms_d(ei - si).count();
            }
            PROGRESS(std::endl);

            // free the strip before the mipmap phase so the two never coexist
            TIMER(timer.start("Free"););
            delete[] rotatedCube;
            rotatedCube = nullptr;

            std::cout << "BENCHMARKING : strip-wise rotation + Z stats: processing " << strip_proc_ms / 1000.0
                      << " s, I/O (swizzled + statsZ writes) " << strip_io_ms / 1000.0 << " s" << std::endl;
            total_pureprocessing_ms += strip_proc_ms;
            total_io_ms += strip_io_ms;
        }

        // ------------------------------------------------------------------ mipmaps in channel blocks
        PROGRESS("\tMipmaps\t\t");
        double mip_proc_ms = 0.0, mip_io_ms = 0.0;
        hsize_t mipBufferChannels = 0; // 0 = no buffers allocated
        for (hsize_t c_start = 0; c_start < depth; c_start += mipChannelBlock) {
            const hsize_t nCh = std::min(mipChannelBlock, depth - c_start);
            PROGRESS("|");

            TIMER(timer.start("Mipmaps"););
            if (nCh != mipBufferChannels) {
                // MipMap::createBuffers frees previous buffers and zeroes the new ones
                mipMaps.createBuffers({nCh, height, width});
                mipBufferChannels = nCh;
            }

            start1 = std::chrono::high_resolution_clock::now();
            if (!zMips) {
                // (channel, row band) pairs touch disjoint mip cells -> race-free
#pragma omp parallel for collapse(2) schedule(dynamic)
                for (hsize_t c = 0; c < nCh; c++) {
                    for (hsize_t b = 0; b < nBands; b++) {
                        const hsize_t yEnd = std::min((b + 1) * maxMipXY, height);
                        const float* channel = standardCube + (c_start + c) * planeSize;
                        for (hsize_t y = b * maxMipXY; y < yEnd; y++) {
                            for (hsize_t x = 0; x < width; x++) {
                                const float val = channel[x + width * y];
                                if (std::isfinite(val)) {
                                    mipMaps.accumulate(val, x, y, c);
                                }
                            }
                        }
                    }
                }
            } else {
                // Z-mips: channels share mip cells, so only row bands are parallel (channels inside a band)
#pragma omp parallel for schedule(dynamic)
                for (hsize_t b = 0; b < nBands; b++) {
                    const hsize_t yEnd = std::min((b + 1) * maxMipXY, height);
                    for (hsize_t c = 0; c < nCh; c++) {
                        const float* channel = standardCube + (c_start + c) * planeSize;
                        for (hsize_t y = b * maxMipXY; y < yEnd; y++) {
                            for (hsize_t x = 0; x < width; x++) {
                                const float val = channel[x + width * y];
                                if (std::isfinite(val)) {
                                    mipMaps.accumulate(val, x, y, c);
                                }
                            }
                        }
                    }
                }
            }
            mipMaps.calculate();
            end1 = std::chrono::high_resolution_clock::now();
            mip_proc_ms += ms_d(end1 - start1).count();

            TIMER(timer.start("Write"););
            start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(currentStokes, c_start);
            end_io = std::chrono::high_resolution_clock::now();
            mip_io_ms += ms_d(end_io - start_io).count();

            TIMER(timer.start("Mipmaps"););
            mipMaps.resetBuffers();
        }
        // free mipmap buffers so the next Stokes' rotation strip can reuse the memory
        TIMER(timer.start("Free"););
        mipMaps.freeBuffers();
        PROGRESS(std::endl);

        std::cout << "BENCHMARKING : mipmaps (" << ((depth + mipChannelBlock - 1) / mipChannelBlock) << " block(s)): processing "
                  << mip_proc_ms / 1000.0 << " s, I/O " << mip_io_ms / 1000.0 << " s" << std::endl;
        total_pureprocessing_ms += mip_proc_ms;
        total_io_ms += mip_io_ms;

        // ------------------------------------------------------------------ write stats
        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats" << std::endl);
        start_io = std::chrono::high_resolution_clock::now();
        statsXY.write({1, depth}, {currentStokes, 0});
        if (depth > 1) {
            statsXYZ.write({1}, {currentStokes});
            // statsZ already written strip by strip
        }
        end_io = std::chrono::high_resolution_clock::now();
        total_io_ms += ms_d(end_io - start_io).count();
    } // end of Stokes loop

    std::cout << "BENCHMARKING: total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds " << float(total_io_ms)/1000.00 << " seconds" << std::endl;

    TIMER(timer.start("Free"););
    delete[] standardCube;
    standardCube = nullptr;
    if (rotatedCube) {
        delete[] rotatedCube;
        rotatedCube = nullptr;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = ms_d(end - start).count();
    std::cout << "Execution of entire FastConverterLimitedMemory::copyAndCalculate took " << duration_ms << " milliseconds " << duration_ms/1000.00 << " seconds" << std::endl;
    std::cout << "Unaccounted for: " << (duration_ms - total_pureprocessing_ms - total_io_ms)/1000.00 << " seconds" << std::endl;
}

// I/O cost model -- like FastConverter::estimateIO except that the swizzled dataset and statsZ are written
// once per strip ({1, nx, H, D} and {1, H, nx}) and the mipmaps once per channel block.
IOCostBreakdown FastConverterLimitedMemory::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                                       hsize_t numBins,
                                                       const IOCostModel& readModel,
                                                       const IOCostModel& writeModel) {
    IOCostBreakdown result;

    const hsize_t sw = std::max((hsize_t)1, std::min(stripWidth, width));

    const std::vector<hsize_t> standardDims4 = {stokes, depth, height, width};
    const std::vector<hsize_t> swizzledDims4 = {stokes, width, height, depth};
    const std::vector<hsize_t> statsZDims    = {stokes, height, width};

    const std::vector<hsize_t> standardChunks =
        useChunks({height, width}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE} : std::vector<hsize_t>{};
    // contiguous: Converter::convert() only chunks the swizzled dataset (-C) for SMART* and SLOW
    const std::vector<hsize_t> swizzledChunks = {};
    const std::vector<hsize_t> statsZChunks = {1, std::min((hsize_t)TILE_SIZE, height), std::min((hsize_t)TILE_SIZE, width)};

    const hsize_t basicElemSizes[] = {4, 4, 4, 4, 8};
    const hsize_t cubeBytes = depth * height * width * sizeof(float);

    {
        PhaseAccumulator acc;
        acc.add(repeatEstimate(IOOpEstimate{1, cubeBytes, cubeBytes}, stokes), readModel);
        result.phases.push_back(acc.toPhase("FITS read (whole Stokes cube)"));
    }
    {
        PhaseAccumulator acc;
        auto e = estimateHyperslabIO(standardDims4, standardChunks, {1, depth, height, width}, sizeof(float));
        acc.add(repeatEstimate(e, stokes), writeModel);
        result.phases.push_back(acc.toPhase("standardDataSet write"));
    }
    if (depth > 1) {
        PhaseAccumulator swzAcc, zAcc;
        for (hsize_t x0 = 0; x0 < width; x0 += sw) {
            hsize_t nx = std::min(sw, width - x0);
            auto w = estimateHyperslabIO(swizzledDims4, swizzledChunks, {1, nx, height, depth}, sizeof(float));
            swzAcc.add(repeatEstimate(w, stokes), writeModel);
            for (auto es : basicElemSizes) {
                auto z = estimateHyperslabIO(statsZDims, statsZChunks, {1, height, nx}, es);
                zAcc.add(repeatEstimate(z, stokes), writeModel);
            }
        }
        result.phases.push_back(swzAcc.toPhase("swizzledDataSet strip writes (contiguous)"));
        result.phases.push_back(zAcc.toPhase("statsZ strip writes"));
    }
    {
        // one write per XY level per channel block (NOTE: Z-mip levels, -z, are not modelled)
        const hsize_t cb = std::max((hsize_t)1, std::min(mipChannelBlock, depth));
        PhaseAccumulator acc;
        for (hsize_t c0 = 0; c0 < depth; c0 += cb) {
            const hsize_t nCh = std::min(cb, depth - c0);
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
        result.phases.push_back(acc.toPhase("mipmap block writes (all XY levels)"));
    }
    {
        PhaseAccumulator acc;
        for (auto es : basicElemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, sizeof(int64_t));
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("statsXY write (basic + channel histograms)"));
    }
    if (depth > 1) {
        PhaseAccumulator acc;
        for (auto es : basicElemSizes) {
            auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, sizeof(int64_t));
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("statsXYZ write (basic + cube histogram)"));
    }

    return result;
}
