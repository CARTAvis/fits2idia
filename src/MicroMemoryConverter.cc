/* This file is part of the FITS to IDIA file format converter: https://github.com/idia-astro/fits2idia
   Copyright 2019, 2020, 2021 the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// ---------------------------------------------------------------------------------------------
// STEP 3: parallelize the main stats+mipmap pass across column-chunks of each row using OpenMP.
//
// Safety note on mipmaps: MipMap::accumulate() bins spatially as
//   mipIndex = (y/mipXY)*width + (x/mipXY)
// with every active mipXY a power of two (2,4,8,...). Column-chunk boundaries below are always
// rounded to a multiple of the *coarsest* mipXY in use, which is automatically a multiple of every
// smaller power-of-two mipXY too — so no two chunks can ever write to the same mip bin, at any
// level. Because of that, each thread can write directly into the shared master `mipMaps` object:
// no per-thread mipmap buffers, no critical section, unlike SmartConverter's approach (which needs
// them because it parallelizes across whole channels/tiles with no such alignment guarantee). This
// keeps memory flat, which matters for a converter whose whole point is a minimal memory footprint.
//
// StatsCounter (min/max/sum/sumSq/nanCount) has no such spatial locality, so each chunk still gets
// its own counter, merged into the channel's counterXY afterward via accumulateFromCounter — cheap,
// since it's a handful of scalars per chunk, not a channel-sized buffer.
//
// The second (histogram) pass is NOT parallelized here: histogram bins are value-based, not
// spatial, so two column-chunks can easily hit the same bin — that needs the partial-histogram
// mechanism already used elsewhere in the codebase (Stats::accumulatePartialHistogram /
// consolidatePartialHistogram), which is a reasonable follow-up step but a distinct piece of work.
// The second (histogram) pass described in STEP 2 no longer exists as a separate reread of the
// FITS file — see STEP 4 below, which folds it into the rotation pass instead.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// STEP 4: fold histogram calculation into the rotation pass instead of a separate reread of the
// FITS file. The rotation pass already streams the whole cube through in bounded TILE_SIZE x
// TILE_SIZE x depth chunks (reading from the already-written HDF5 standardDataSet), and histogram
// binning is order-independent — it doesn't matter that this pass visits pixels tile-by-tile
// rather than row-by-row. So the channel (XY) and cube (XYZ) histograms are now accumulated
// alongside the existing Z-statistics loop in the rotation pass, using the exact same value reads.
//
// Only applies when depth > 1, since the swizzled dataset (and hence this whole rotation pass)
// only exists then. For depth == 1 there's no rotation pass to fold into, so the single channel's
// histogram is still computed in its own small pass right after pass 1, same as before.
//
// One wrinkle: statsXY/statsXYZ are single buffers reused across all stokes values, so by the time
// the rotation pass's own stokes loop reaches a given stokes, pass 1 has already moved on and
// overwritten those buffers with a later stokes' values. Two small fixes for that:
//   - a tiny cache (savedChanMin/Max, savedCubeMin/Max — a few doubles per stokes/channel) records
//     each stokes' min/max right after pass 1 computes it, for the rotation pass to read back later.
//   - writing to disk is split into writeBasic() (called immediately after pass 1, while still
//     correct) and writeHistogram() (called later, once the rotation pass has filled it in) instead
//     of the combined write() — otherwise write() would write the wrong (later) stokes' basic stats
//     for every stokes except the last one.
//
// NOTE ruled out: folding the *rotation* itself into pass 1, the way SmartFastConverter does, was
// considered and deliberately not done — it requires a persistent per-pixel Z-statistics buffer
// (height x width x sizeof(StatsCounter)) to carry state across blocks, which is far larger than
// even a single channel and defeats this class's whole reason to exist.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// STEP 5: pass 1 reads/writes a BLOCK of rows_per_block image rows per call instead of one row.
//
// rows_per_block is sized from the memory limit (setMemoryLimit / ReduceMemoryUsage -> fitToMemory):
//   - enough memory for a whole channel -> rows_per_block = height, i.e. exactly the per-channel I/O
//     of the old SlowConverter (but still without its second FITS reread, see STEP 4);
//   - otherwise the largest multiple of TILE_SIZE rows that fits, so every block covers whole rows of
//     standardDataSet chunks ({1,1,TILE_SIZE,TILE_SIZE}) and each chunk is written in one piece instead
//     of TILE_SIZE partial row writes;
//   - below TILE_SIZE rows, the largest power of two (a divisor of TILE_SIZE, so blocks never straddle a
//     chunk row); 1 row is the old behaviour of this converter.
// The FITS read needs no new helper: readFitsDataRow() starts at (row y0) and fits_read_pix() reads
// nRows*width consecutive pixels, which are exactly rows y0..y0+nRows-1 of that channel.
//
// The row-block buffer and the rotation-pass tile buffers never coexist, so both knobs (rows_per_block,
// rotation_tile_size) are fitted independently against the same budget:
//   peak = fixed (mipmaps + XY/XYZ stats) + max(row block, rotation tile buffers)
// ---------------------------------------------------------------------------------------------

#include "Converter.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

MicroMemoryConverter::MicroMemoryConverter(std::string inputFileName, std::string outputFileName, bool progress, bool zMips)
    : SmartConverter(inputFileName, outputFileName, progress, zMips), rows_per_block(0) {
    // default (no memory limit): whole channel per pass-1 call, full TILE_SIZE rotation tiles
    rows_per_block = height;
    rotation_tile_size = TILE_SIZE;
}

// Alive for the whole conversion: mipmap buffers (one channel) and the XY / XYZ stats buffers.
hsize_t MicroMemoryConverter::fixedMemory() {
    hsize_t fixed = MipMaps::size(standardDims, {1, height, width}, zMips);
    fixed += Stats::size({depth}, numBins);
    if (depth > 1) {
        fixed += Stats::size({}, numBins, depth);
    }
    return fixed;
}

// Pass-1 buffer: rows image rows of one channel.
hsize_t MicroMemoryConverter::rowBlockMemory(hsize_t rows) {
    return rows * width * sizeof(float);
}

// Rotation pass: standardSlice + rotatedSlice (one Stokes, full depth, tile x tile) + tile-sized Z stats.
hsize_t MicroMemoryConverter::rotationMemory(hsize_t tile) {
    if (depth <= 1) {
        return 0;
    }
    return 2 * product(trimAxes({1, depth, tile, tile}, N)) * sizeof(float) + Stats::size({tile, tile});
}

// Largest row count whose buffer fits in budget, aligned to the standardDataSet chunk rows:
// whole channel if possible, else a multiple of TILE_SIZE, else a power of two (divides TILE_SIZE).
// Returns 0 if not even one row fits.
hsize_t MicroMemoryConverter::chooseRowsPerBlock(hsize_t budget) {
    const hsize_t maxRows = budget / (width * sizeof(float));
    if (maxRows == 0) {
        return 0;
    }
    if (maxRows >= height) {
        return height;
    }
    if (maxRows >= TILE_SIZE) {
        return (maxRows / TILE_SIZE) * TILE_SIZE;
    }
    hsize_t rows = 1;
    while (rows * 2 <= maxRows) {
        rows *= 2;
    }
    return rows;
}

// Sets rows_per_block and rotation_tile_size for the given limit. Leaves both untouched and returns
// false if nothing fits (so a failed attempt never degrades the configuration).
bool MicroMemoryConverter::fitToMemory(hsize_t memoryLimit) {
    const hsize_t fixed = fixedMemory();
    if (fixed >= memoryLimit) {
        std::cout << "MicroMemoryConverter: cannot fit, mipmaps + XY/XYZ stats alone need " << fixed * 1e-9
                  << " GB (limit " << memoryLimit * 1e-9 << " GB)" << std::endl;
        return false;
    }
    const hsize_t budget = memoryLimit - fixed;

    const hsize_t rows = chooseRowsPerBlock(budget);
    if (rows == 0) {
        std::cout << "MicroMemoryConverter: cannot fit, not even one image row (" << rowBlockMemory(1) * 1e-9
                  << " GB) fits in the remaining " << budget * 1e-9 << " GB" << std::endl;
        return false;
    }

    hsize_t tile = TILE_SIZE;
    if (depth > 1) {
        while (rotationMemory(tile) > budget && tile / 2 >= (hsize_t)MIN_ROTATION_TILE) {
            tile /= 2;
        }
        if (rotationMemory(tile) > budget) {
            std::cout << "MicroMemoryConverter: cannot fit, even " << tile << "x" << tile << " rotation tiles need "
                      << rotationMemory(tile) * 1e-9 << " GB (remaining budget " << budget * 1e-9 << " GB)" << std::endl;
            return false;
        }
    }

    rows_per_block = rows;
    rotation_tile_size = (int)tile;
    std::cout << "MicroMemoryConverter: rows_per_block = " << rows_per_block << " (of " << height
              << "), rotation_tile_size = " << rotation_tile_size << std::endl;
    return true;
}

void MicroMemoryConverter::setMemoryLimit(int _memoryLimitInMb) {
    memoryLimitInMb = _memoryLimitInMb;
    if (memoryLimitInMb > 0) {
        if (!fitToMemory((hsize_t)(memoryLimitInMb * 1e6))) { // same MB convention as main.cc
            std::cout << "WARNING (MicroMemoryConverter) : configuration does not fit in " << memoryLimitInMb
                      << " MB -> keeping rows_per_block = " << rows_per_block << ", rotation_tile_size = "
                      << rotation_tile_size << " (checkMemoryUsage will report the overrun)" << std::endl;
        }
    } else {
        rows_per_block = height;
        rotation_tile_size = TILE_SIZE;
    }
}

MemoryUsage MicroMemoryConverter::calculateMemoryUsage() {
    MemoryUsage m;
    const hsize_t T = rotation_tile_size;

    m.sizes["Main dataset (block of " + std::to_string(rows_per_block) + " rows)"] = rowBlockMemory(rows_per_block);
    m.sizes["Mipmaps"] = MipMaps::size(standardDims, {1, height, width}, zMips);
    m.sizes["XY stats"] = Stats::size({depth}, numBins);

    if (depth > 1) {
        m.sizes["Rotation (" + std::to_string(T) + "x" + std::to_string(T) + " tiles)"] = 2 * product(trimAxes({1, depth, T, T}, N)) * sizeof(float);
        m.sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        m.sizes["Z stats"] = Stats::size({T, T});
    }

    // the row-block buffer and the rotation-pass buffers are never allocated at the same time
    m.total = fixedMemory() + std::max(rowBlockMemory(rows_per_block), rotationMemory(T));
    m.note = " (row-block buffer and rotation/Z-stats tile buffers are not allocated at the same time)";

    DEBUG(printf("MicroMemoryConverter::calculateMemoryUsage = %.3f [MB]\n", m.total / 1e6););

    return m;
}

bool MicroMemoryConverter::ReduceMemoryUsage(hsize_t memoryLimit, int max_iter) {
    UNUSED(max_iter);
    if (calculateMemoryUsage().total <= memoryLimit) {
        return true;
    }
    return fitToMemory(memoryLimit);
}

void MicroMemoryConverter::copyAndCalculate() {
    auto start = std::chrono::high_resolution_clock::now();
    const hsize_t T = rotation_tile_size;
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));
    hsize_t numTiles = ((width + T - 1) / T) * ((height + T - 1) / T);
    const hsize_t tileProgressStride = std::max((hsize_t)1, (hsize_t)(numTiles / 100));

    // STEP 2/5: allocate a block of rows_per_block rows (all X, consecutive Y, one channel, one stokes).
    const hsize_t R = std::max((hsize_t)1, std::min(rows_per_block, height));
    TIMER(timer.start("Allocate"););
    standardCube = new float[R * width];
    std::cout << "MicroMemoryConverter: rows_per_block = " << R << " -> " << (height + R - 1) / R
              << " FITS read / HDF5 write call(s) per channel" << std::endl;

    // Allocate one stokes of stats at a time
    statsXY.createBuffers({depth});

    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
    }

    mipMaps.createBuffers({1, height, width});

    // STEP 3: work out column-chunk boundaries for the row-parallel loop below, aligned to the
    // coarsest mip factor in use so that concurrent chunks can never write to the same mip bin.
    int coarsestMipXY = 1;
    for (auto& mipMap : mipMaps.mipMaps) {
        coarsestMipXY = std::max(coarsestMipXY, mipMap.mipXY);
    }

    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    std::cout << "MicroMemoryConverter: numThreads = " << numThreads << std::endl;

    hsize_t chunkWidth = (hsize_t)coarsestMipXY * std::max((hsize_t)1, (width / numThreads) / (hsize_t)coarsestMipXY);

    std::vector<std::pair<hsize_t, hsize_t>> columnChunks; // [start, end) pairs
    for (hsize_t xStart = 0; xStart < width; xStart += chunkWidth) {
        columnChunks.push_back({xStart, std::min(width, xStart + chunkWidth)});
    }
    std::cout << "MicroMemoryConverter: splitting each row of width " << width << " into " << columnChunks.size()
              << " column chunks of ~" << chunkWidth << " columns (coarsest mip factor = " << coarsestMipXY << ")" << std::endl;

    // One StatsCounter scratch slot per chunk, reused (and reset) for every row block.
    std::vector<StatsCounter> chunkCounters(columnChunks.size());

    std::string timerLabelStatsMipmaps = depth > 1 ? "XY and XYZ statistics and mipmaps" : "XY statistics and mipmaps";

    // STEP 4: for depth > 1, the histogram calculation moves into the rotation pass below (it
    // already streams through the whole cube in bounded tiles, so no separate reread of the FITS
    // file is needed). That pass has its own stokes loop that only runs after this per-stokes loop
    // has finished ALL stokes — by which point statsXY/statsXYZ's in-memory buffers hold only the
    // LAST stokes' values (they're a single reused buffer, overwritten every stokes). This tiny
    // cache — a handful of doubles per stokes/channel — is all that's needed to remember the rest.
    std::vector<double> savedChanMin(stokes * depth), savedChanMax(stokes * depth);
    std::vector<double> savedCubeMin(stokes), savedCubeMax(stokes);

    double total_io_ms = 0.00, total_pureprocessing_ms = 0.00;
    for (unsigned int s = 0; s < stokes; s++) {
        DEBUG(std::cout << "Processing Stokes " << s << "... " << std::endl;);
        PROGRESS("Stokes " << s << ":" << std::endl);

        PROGRESS("\tMain loop\t");

        StatsCounter counterXYZ;

        double total_first_pass_processing_ms = 0.00;
        for (hsize_t c = 0; c < depth; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            DEBUG(std::cout << "+ Processing channel " << c << " row by row... " << std::flush;);

            DEBUG(std::cout << " Accumulating XY stats and mipmaps..." << std::flush;);

            StatsCounter counterXY;
            auto indexXY = c;

            double channel_io_ms = 0.00;
            auto start1 = std::chrono::high_resolution_clock::now();
            for (hsize_t y0 = 0; y0 < height; y0 += R) {
                const hsize_t nRows = std::min(R, height - y0);

                // read nRows consecutive rows: fits_read_pix continues across row ends
                auto start_io = std::chrono::high_resolution_clock::now();
                TIMER(timer.start("Read"););
                readFitsDataRow(inputFilePtr, c, y0, s, nRows * width, standardCube, swapStokesFreqAxis);
                auto end_io = std::chrono::high_resolution_clock::now();
                channel_io_ms += double(ms_d(end_io - start_io).count());

                // write the row block to the standard dataset (whole chunk rows when R % TILE_SIZE == 0)
                TIMER(timer.start("Write"););
                start_io = std::chrono::high_resolution_clock::now();
                std::vector<hsize_t> blockMemDims = {nRows, width};
                std::vector<hsize_t> blockCount = trimAxes({1, 1, nRows, width}, N);
                std::vector<hsize_t> blockStart = trimAxes({s, c, y0, 0}, N);
                writeHdf5Data(standardDataSet, standardCube, blockMemDims, blockCount, blockStart);
                end_io = std::chrono::high_resolution_clock::now();
                channel_io_ms += double(ms_d(end_io - start_io).count());

                TIMER(timer.start(timerLabelStatsMipmaps););

                // STEP 3: process the block's columns in parallel chunks (each chunk covers all nRows rows).
                // Each chunk accumulates into its own StatsCounter (merged below) and writes directly into
                // the shared mipMaps object -- safe because chunk boundaries are aligned to the coarsest mip
                // factor, so disjoint column ranges never share a mip cell (see note at top of file).
                for (auto& cc : chunkCounters) {
                    cc.reset();
                }

#pragma omp parallel for schedule(dynamic) shared(columnChunks, chunkCounters)
                for (size_t ci = 0; ci < columnChunks.size(); ci++) {
                    const hsize_t xStart = columnChunks[ci].first;
                    const hsize_t xEnd = columnChunks[ci].second;
                    StatsCounter& counterChunk = chunkCounters[ci];

                    std::function<void(float)> accumulateChunk;

                    auto lazy_accumulate_chunk = [&] (float val) {
                        counterChunk.accumulateFiniteLazy(val);
                    };

                    auto first_accumulate_chunk = [&] (float val) {
                        counterChunk.accumulateFiniteLazyFirst(val);
                        accumulateChunk = lazy_accumulate_chunk;
                    };

                    accumulateChunk = first_accumulate_chunk;

                    for (hsize_t r = 0; r < nRows; r++) {
                        const hsize_t y = y0 + r;              // GLOBAL row index for the mipmaps
                        const float* row = standardCube + r * width;
                        for (hsize_t x = xStart; x < xEnd; x++) {
                            const float val = row[x];

                            if (std::isfinite(val)) {
                                accumulateChunk(val);
                                mipMaps.accumulate(val, x, y, 0);
                            } else {
                                counterChunk.accumulateNonFinite();
                            }
                        }
                    }
                } // end of parallel column-chunk loop

                for (auto& counterChunk : chunkCounters) {
                    counterXY.accumulateFromCounter(counterChunk);
                }
            } // end of row-block loop
            total_io_ms += channel_io_ms;
            std::cout << "I/O (readFitsDataRow+writeHdf5Data, " << (height + R - 1) / R << " block(s) of up to " << R << " rows) for channel : " << c << " took " << channel_io_ms << " milliseconds." << std::endl;

            // Final correction of XY min and max
            DEBUG(std::cout << " Final XY stats..." << std::flush;);
            statsXY.copyStatsFromCounter(indexXY, height * width, counterXY);

            // Accumulate XYZ statistics
            if (depth > 1) {
                DEBUG(std::cout << " Accumulating XYZ stats..." << std::flush;);
                statsXY.accumulateStatsToCounter(counterXYZ, indexXY);
            }

            // Final mipmap calculation
            DEBUG(std::cout << " Final mipmaps..." << std::flush;);
            mipMaps.calculate();

            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = ms_d(end1 - start1);
            std::cout << "Execution of 1st loop (row I/O + stats/mipmap accumulation) for channel " << c << " took: " << duration1.count() << " milliseconds." << std::endl;
            total_first_pass_processing_ms += double(duration1.count()) - channel_io_ms;

            // Write the mipmaps
            DEBUG(std::cout << " Writing mipmaps..." << std::flush;);
            TIMER(timer.start("Write"););
            auto start_io = std::chrono::high_resolution_clock::now();
            mipMaps.write(s, c);
            auto end_io = std::chrono::high_resolution_clock::now();
            total_io_ms += double(ms_d(end_io - start_io).count());

            // Reset mipmaps before next channel
            DEBUG(std::cout << " Resetting mipmap objects..." << std::endl;);
            TIMER(timer.start(timerLabelStatsMipmaps););
            mipMaps.resetBuffers();

        } // end of first channel loop
        std::cout << "BENCHMARKING : total pure-processing time of 1st pass: " << total_first_pass_processing_ms << " milliseconds " << float(total_first_pass_processing_ms)/1000.00 << " seconds" << std::endl;
        total_pureprocessing_ms += total_first_pass_processing_ms;

        PROGRESS(std::endl);

        if (depth > 1) {
            // Final correction of XYZ min and max
            DEBUG(std::cout << " Final XYZ stats..." << std::flush;);
            PROGRESS("\tXYZ stats" << std::endl);
            TIMER(timer.start(timerLabelStatsMipmaps););
            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);
        }

        if (depth == 1) {
            // STEP 4: with only one channel there's no rotation pass to fold this into (the
            // swizzled dataset only exists for depth > 1), so we still need a dedicated pass here
            // to fill in the channel histogram — we don't know this channel's min/max until it's
            // been fully read once in the main loop above.
            DEBUG(std::cout << " Histogram (single channel)..." << std::endl;);
            PROGRESS("\tHistogram\t");
            TIMER(timer.start("Histograms"););

            statsXY.clearHistogramBuffers();

            hsize_t c = 0;
            double chanMin = statsXY.minVals[c];
            double chanMax = statsXY.maxVals[c];
            double chanRange = chanMax - chanMin;
            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
            DEBUG(std::cout << " Will " << (chanHist ? "" : "not ") << "calculate channel histogram." << std::flush;);

            if (chanHist) {
                double channel_io_ms = 0.00;
                auto start1 = std::chrono::high_resolution_clock::now();
                for (hsize_t y0 = 0; y0 < height; y0 += R) {
                    const hsize_t nRows = std::min(R, height - y0);
                    auto start_io = std::chrono::high_resolution_clock::now();
                    TIMER(timer.start("Read"););
                    readFitsDataRow(inputFilePtr, c, y0, s, nRows * width, standardCube, swapStokesFreqAxis);
                    auto end_io = std::chrono::high_resolution_clock::now();
                    channel_io_ms += double(ms_d(end_io - start_io).count());

                    TIMER(timer.start("Histograms"););
                    for (hsize_t p = 0; p < nRows * width; p++) {
                        auto& val = standardCube[p];
                        if (std::isfinite(val)) {
                            statsXY.accumulateHistogram(val, chanMin, chanRange, c);
                        }
                    }
                }
                total_io_ms += channel_io_ms;
                auto end1 = std::chrono::high_resolution_clock::now();
                auto duration1 = ms_d(end1 - start1);
                total_pureprocessing_ms += double(duration1.count()) - channel_io_ms;
                std::cout << "Execution of single-channel histogram pass took: " << duration1.count() << " milliseconds." << std::endl;
            }

            PROGRESS(std::endl);

            // Write the statistics
            TIMER(timer.start("Write"););
            PROGRESS("\tWrite stats & mipmaps" << std::endl);

            auto start_io = std::chrono::high_resolution_clock::now();
            statsXY.write({1, depth}, {s, 0});
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = ms_d(end_io - start_io);
            total_io_ms += double(duration_io.count());
        } else {
            // STEP 4: depth > 1 — defer histogram calculation to the rotation pass below, which
            // already streams the whole cube through in TILE_SIZE x TILE_SIZE x depth chunks.
            // Reusing that read means no separate reread of the FITS file for histograms at all.
            //
            // Basic stats (min/max/sum/sumSq/nanCount) ARE already correct right now, so write
            // those immediately — writeBasic() only, not write(), since statsXY/statsXYZ are
            // single reused buffers: by the time the rotation pass gets around to this stokes, the
            // buffer will hold a LATER stokes' basic values, so it can't be the source for those at
            // that point. The histogram half is written separately, later, once it's been filled in.
            auto start_io = std::chrono::high_resolution_clock::now();
            {
                auto basicN = statsXY.basicDatasetDims.size();
                statsXY.writeBasic(statsXY.fullBasicBufferDims, trimAxes({1, depth}, basicN), trimAxes({s, 0}, basicN));
            }
            {
                auto basicN = statsXYZ.basicDatasetDims.size();
                statsXYZ.writeBasic(statsXYZ.fullBasicBufferDims, trimAxes({1}, basicN), trimAxes({s}, basicN));
            }
            auto end_io = std::chrono::high_resolution_clock::now();
            auto duration_io = ms_d(end_io - start_io);
            total_io_ms += double(duration_io.count());

            // Cache this stokes' channel/cube min-max now, before the next stokes overwrites them.
            for (hsize_t c = 0; c < depth; c++) {
                savedChanMin[s * depth + c] = statsXY.minVals[c];
                savedChanMax[s * depth + c] = statsXY.maxVals[c];
            }
            savedCubeMin[s] = statsXYZ.minVals[0];
            savedCubeMax[s] = statsXYZ.maxVals[0];
        }

    } // end of stokes

    // Free memory
    DEBUG(std::cout << "Freeing memory from row-block buffer... " << std::endl;);
    TIMER(timer.start("Free"););

    delete[] standardCube;

    // Swizzle -- reads the already-written HDF5 standardDataSet in rotation_tile_size x
    // rotation_tile_size x depth tiles, so it doesn't care how many rows pass 1 wrote per call.
    if (depth > 1) {
        DEBUG(std::cout << "Performing tiled rotation." << std::endl;);
        PROGRESS("Tiled rotation & Z stats" << std::endl);
        TIMER(timer.start("Allocate"););
        
        hsize_t sliceSize = product(trimAxes({1, depth, T, T}, N));
        float* standardSlice = new float[sliceSize];
        float* rotatedSlice = new float[sliceSize];

        statsZ.createBuffers({T, T});

        for (unsigned int s = 0; s < stokes; s++) {
            DEBUG(std::cout << "Processing Stokes " << s << "..." << std::endl;);
            PROGRESS("\tStokes " << s << "\t");

            // STEP 4: histograms are computed here instead of in a separate reread of the FITS
            // file — this pass already streams the whole cube through in bounded tiles, and
            // histogram binning doesn't care what order the pixels arrive in. Pull this stokes'
            // saved channel/cube min-max (cached back in the main per-stokes loop, since pass 1's
            // in-memory statsXY/statsXYZ buffers have since moved on to later stokes).
            statsXY.clearHistogramBuffers();
            statsXYZ.clearHistogramBuffers();

            double cubeMin = savedCubeMin[s];
            double cubeMax = savedCubeMax[s];
            double cubeRange = cubeMax - cubeMin;
            bool cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;

            std::vector<double> chanMinArr(depth), chanRangeArr(depth);
            std::vector<bool> chanHistArr(depth);
            for (hsize_t c = 0; c < depth; c++) {
                double chanMin = savedChanMin[s * depth + c];
                double chanMax = savedChanMax[s * depth + c];
                chanMinArr[c] = chanMin;
                chanRangeArr[c] = chanMax - chanMin;
                chanHistArr[c] = std::isfinite(chanMin) && std::isfinite(chanMax) && chanRangeArr[c] > 0;
            }

            DEBUG(std::cout << "+ Will " << (cubeHist ? "" : "not ") << "calculate cube histogram." << std::endl;);

            hsize_t tileCount(0);

            double total_rotation_pass_processing_ms = 0.00;
            auto start1 = std::chrono::high_resolution_clock::now();
            for (hsize_t xOffset = 0; xOffset < width; xOffset += T) {
                for (hsize_t yOffset = 0; yOffset < height; yOffset += T) {
                    auto starttile = std::chrono::high_resolution_clock::now();
                    tileCount++;
                    hsize_t xSize = std::min(T, width - xOffset);
                    hsize_t ySize = std::min(T, height - yOffset);

                    DEBUG(std::cout << "+ Processing tile slice at " << xOffset << ", " << yOffset << "..." << std::flush;);
                    PROGRESS_DECIMATED(tileCount, tileProgressStride, "#");

                    // read tile slice
                    DEBUG(std::cout << " Reading main dataset..." << std::flush;);
                    TIMER(timer.start("Read"););

                    auto standardMemDims = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardCount = trimAxes({1, depth, ySize, xSize}, N);
                    auto standardStart = trimAxes({s, 0, yOffset, xOffset}, N);

                    auto start_io = std::chrono::high_resolution_clock::now();
                    readHdf5Data(standardDataSet, standardSlice, standardMemDims, standardCount, standardStart);
                    auto end_io = std::chrono::high_resolution_clock::now();
                    auto duration_io = ms_d(end_io - start_io);
                    total_io_ms += double(duration_io.count());

                    // rotate tile slice
                    DEBUG(std::cout << " Calculating rotation..." << std::flush;);
                    TIMER(timer.start("Rotation"););

                    auto start2 = std::chrono::high_resolution_clock::now();
                    for (hsize_t i = 0; i < depth; i++) {
                        for (hsize_t j = 0; j < ySize; j++) {
                            for (hsize_t k = 0; k < xSize; k++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto& val = standardSlice[sourceIndex];

                                // rotation
                                auto destIndex = i + depth * j + (ySize * depth) * k;
                                rotatedSlice[destIndex] = val;
                            }
                        }
                    }
                    auto end2 = std::chrono::high_resolution_clock::now();
                    auto duration2 = ms_d(end2 - start2);
                    std::cout << "Execution of small rotation-loop took: " << duration2.count() << " milliseconds." << std::endl;

                    // A separate pass over the same slice depth-last
                    DEBUG(std::cout << " Calculating Z statistics..." << std::flush;);
                    TIMER(timer.start("Z statistics"););

                    auto start3 = std::chrono::high_resolution_clock::now();
                    for (hsize_t j = 0; j < ySize; j++) {
                        for (hsize_t k = 0; k < xSize; k++) {
                            StatsCounter counterZ;
                            auto indexZ = k + xSize * j;

                            for (hsize_t i = 0; i < depth; i++) {
                                auto sourceIndex = k + xSize * j + (ySize * xSize) * i;
                                auto& val = standardSlice[sourceIndex];

                                if (std::isfinite(val)) {
                                    // Not lazy; too much risk of encountering an ascending / descending sequence.
                                    counterZ.accumulateFinite(val);

                                    // STEP 4: channel and cube histograms, using the same value this
                                    // loop already read for Z statistics — no extra pass needed.
                                    if (chanHistArr[i]) {
                                        statsXY.accumulateHistogram(val, chanMinArr[i], chanRangeArr[i], i);
                                    }
                                    if (cubeHist) {
                                        statsXYZ.accumulateHistogram(val, cubeMin, cubeRange, 0);
                                    }
                                } else {
                                    counterZ.accumulateNonFinite();
                                }
                            }

                            statsZ.copyStatsFromCounter(indexZ, depth, counterZ);
                        }
                    }
                    auto end3 = std::chrono::high_resolution_clock::now();
                    auto duration3 = ms_d(end3 - start3);
                    std::cout << "Execution of counter/stats-Z loop took: " << duration3.count() << " milliseconds." << std::endl;
                    auto duration_processing = ms_d(end3 - start2);
                    total_rotation_pass_processing_ms += double(duration_processing.count());

                    // write tile slice
                    DEBUG(std::cout << " Writing rotated dataset..." << std::endl;);
                    TIMER(timer.start("Write"););

                    auto swizzledMemDims = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledCount = trimAxes({1, xSize, ySize, depth}, N);
                    auto swizzledStart = trimAxes({s, xOffset, yOffset, 0}, N);

                    start_io = std::chrono::high_resolution_clock::now();
                    writeHdf5Data(swizzledDataSet, rotatedSlice, swizzledMemDims, swizzledCount, swizzledStart);
                    end_io = std::chrono::high_resolution_clock::now();
                    duration_io = ms_d(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    DEBUG(std::cout << "3rd I/O (writeHdf5Data) for xOffset = " << xOffset << " yOffset = " << yOffset  << " took " << duration_io.count() << " milliseconds." << std::endl;);

                    start_io = std::chrono::high_resolution_clock::now();
                    DEBUG(std::cout << " Writing Z statistics..." << std::endl;);
                    // write Z statistics
                    statsZ.write({ySize, xSize}, {1, ySize, xSize}, {s, yOffset, xOffset});
                    end_io = std::chrono::high_resolution_clock::now();
                    duration_io = ms_d(end_io - start_io);
                    total_io_ms += double(duration_io.count());
                    DEBUG(std::cout << "4th I/O (statsZ.write) for xOffset = " << xOffset << " yOffset = " << yOffset  << " took " << duration_io.count() << " milliseconds." << std::endl;);

                    auto endtile = std::chrono::high_resolution_clock::now();
                    auto durationtile = ms_d(endtile - starttile);
                    std::cout << "Execution of rotation of 1 tile, including writting, took: " << durationtile.count() << " milliseconds." << std::endl;
                }
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            auto duration1 = ms_d(end1 - start1);
            std::cout << "Execution of loop over X/Y Offsets for Stokes = " << s << " took: " << duration1.count() << " milliseconds." << std::endl;
            std::cout << "BENCHMARKING : total pure-processing time of rotation pass: " << total_rotation_pass_processing_ms << " milliseconds "
                      << (float(total_rotation_pass_processing_ms)/1000.0) << " seconds" << std::endl;
            total_pureprocessing_ms += total_rotation_pass_processing_ms;

            // Write the histogram half of the stats now that it's been filled in above — the
            // basic half was already correct and written right after pass 1, before this stokes'
            // slot in the shared statsXY/statsXYZ buffer got overwritten by later stokes.
            TIMER(timer.start("Write"););
            auto start_io = std::chrono::high_resolution_clock::now();
            {
                auto basicN = statsXY.basicDatasetDims.size();
                auto histN = basicN + 1;
                statsXY.writeHistogram(statsXY.fullBasicBufferDims, trimAxes(extend({1, depth}, {statsXY.numBins}), histN), trimAxes(extend({s, 0}, {0}), histN));
            }
            {
                auto basicN = statsXYZ.basicDatasetDims.size();
                auto histN = basicN + 1;
                statsXYZ.writeHistogram(statsXYZ.fullBasicBufferDims, trimAxes(extend({1}, {statsXYZ.numBins}), histN), trimAxes(extend({s}, {0}), histN));
            }
            auto end_io = std::chrono::high_resolution_clock::now();
            total_io_ms += double(ms_d(end_io - start_io).count());

            PROGRESS(std::endl);
        }

        TIMER(timer.start("Free"););
        DEBUG(std::cout << "Freeing memory from main and rotated dataset slices... " << std::endl;);
        delete[] standardSlice;
        delete[] rotatedSlice;
    }

    std::cout << "BENCHMARKING: Total time spent in I/O (both read and write) = " << total_io_ms << " milliseconds." << std::endl;
    std::cout << "BENCHMARKING: Total pure-processing time of 1st, 2nd and rotation passes: " << total_pureprocessing_ms << " milliseconds " <<  (float(total_pureprocessing_ms)/1000.00) << " seconds" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = ms_d(end - start);
    std::cout << "Execution of entire MicroMemoryConverter::copyAndCalculate took: " << duration.count() << " milliseconds = " << duration.count()/1000.00 << " seconds." << std::endl;
    double unaccounted_for_ms = duration.count() - total_pureprocessing_ms - total_io_ms;
    std::cout << "Unaccounted for: " << unaccounted_for_ms/1000.00 << " seconds" << std::endl;

}

IOCostBreakdown MicroMemoryConverter::estimateIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width, hsize_t numBins, const IOCostModel& readModel, const IOCostModel& writeModel) 
{
    IOCostBreakdown result;

    std::vector<hsize_t> standardDims = {stokes, depth, height, width};
    // standardDataSet: chunked {1,1,TILE,TILE} only if both image dims >= TILE_SIZE (as in Converter::convert)
    std::vector<hsize_t> chunkDims = useChunks({height, width}) ? std::vector<hsize_t>{1, 1, TILE_SIZE, TILE_SIZE}
                                                                : std::vector<hsize_t>{};

    // ---------- Pass 1: one FITS read + one HDF5 write per block of R rows ----------
    // R = rows_per_block: R == height reproduces the old SlowConverter per-channel I/O, R == 1 the old
    // row-at-a-time behaviour. With R a multiple of TILE_SIZE every write covers whole chunk rows.
    const hsize_t R = std::max((hsize_t)1, std::min(rows_per_block, height));
    const hsize_t nFullBlocks = height / R;
    const hsize_t lastRows = height % R;

    {
        PhaseAccumulator acc;
        hsize_t bytesPerOp = R * width * sizeof(float);
        acc.add(repeatEstimate(IOOpEstimate{1, bytesPerOp, bytesPerOp}, stokes * depth * nFullBlocks), readModel);
        if (lastRows > 0) {
            hsize_t lastBytes = lastRows * width * sizeof(float);
            acc.add(repeatEstimate(IOOpEstimate{1, lastBytes, lastBytes}, stokes * depth), readModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: FITS row-block read (" + std::to_string(R) + " rows)"));
    }
    {
        PhaseAccumulator acc;
        auto perBlock = estimateHyperslabIO(standardDims, chunkDims, {1, 1, R, width}, sizeof(float));
        acc.add(repeatEstimate(perBlock, stokes * depth * nFullBlocks), writeModel);
        if (lastRows > 0) {
            auto lastBlock = estimateHyperslabIO(standardDims, chunkDims, {1, 1, lastRows, width}, sizeof(float));
            acc.add(repeatEstimate(lastBlock, stokes * depth), writeModel);
        }
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet row-block write (" + std::to_string(R) + " rows)"));
    }

    // mipMaps.write(s, c): still once per channel (the row loop only computes one
    // channel's mip contribution before this write happens) -- same shape and cost
    // as SlowConverter's mipmap-write phase.
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

    if (depth <= 1) {
        // ---------- depth == 1: single-channel histogram pass ----------
        // FITS reread of the single channel in the same row blocks as pass 1.
        PhaseAccumulator acc;
        hsize_t bytesPerOp = R * width * sizeof(float);
        acc.add(repeatEstimate(IOOpEstimate{1, bytesPerOp, bytesPerOp}, stokes * nFullBlocks), readModel);
        if (lastRows > 0) {
            hsize_t lastBytes = lastRows * width * sizeof(float);
            acc.add(repeatEstimate(IOOpEstimate{1, lastBytes, lastBytes}, stokes), readModel);
        }
        result.phases.push_back(acc.toPhase("Histogram pass (depth==1): FITS row-block reread"));

        PhaseAccumulator statsAcc;
        const hsize_t elemSizes[] = {4, 4, 4, 4, 8}; // MIN, MAX, SUM, SUM_SQ (float on disk), NAN_COUNT (int64)
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            statsAcc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, 8);
            statsAcc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(statsAcc.toPhase("Histogram pass (depth==1): statsXY write"));
    } else {
        // ---------- depth > 1: writeBasic()/writeHistogram() split, lines ~356-372, ~552-563 ----------
        // Two separate small metadata calls instead of SlowConverter's one combined
        // write() -- but each is still 1 transaction (same reasoning as SlowConverter's
        // statsXY write), so the total cost here is essentially the same as
        // SlowConverter's statsXY+statsXYZ phase, just split into more, smaller calls.
        PhaseAccumulator basicAcc;
        {
            const hsize_t elemSizes[] = {4, 4, 4, 4, 8}; // MIN, MAX, SUM, SUM_SQ (float on disk), NAN_COUNT (int64)
            for (auto es : elemSizes) {
                auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
                basicAcc.add(repeatEstimate(e, stokes), writeModel);
            }
        }
        {
            const hsize_t elemSizes[] = {4, 4, 4, 4, 8}; // MIN, MAX, SUM, SUM_SQ (float on disk), NAN_COUNT (int64)
            for (auto es : elemSizes) {
                auto e = estimateHyperslabIO({stokes}, {}, {1}, es);
                basicAcc.add(repeatEstimate(e, stokes), writeModel);
            }
        }
        result.phases.push_back(basicAcc.toPhase("1st pass: statsXY/statsXYZ writeBasic"));

        if (numBins > 0) {
            PhaseAccumulator histAcc;
            auto hXY = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, 8);
            histAcc.add(repeatEstimate(hXY, stokes), writeModel);
            auto hXYZ = estimateHyperslabIO({stokes, numBins}, {}, {1, numBins}, 8);
            histAcc.add(repeatEstimate(hXYZ, stokes), writeModel);
            result.phases.push_back(histAcc.toPhase("Rotation pass: statsXY/statsXYZ writeHistogram"));
        }

        // ---------- Tiled rotation pass (rotation_tile_size tiles) ----------
        addTiledRotationPhases(result, stokes, depth, height, width, readModel, writeModel, getSwizzledChunkDims(), rotation_tile_size);
    }

    return result;
}
