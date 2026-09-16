#include "IOCost.h"
#include "Util.h"



IOOpEstimate estimateHyperslabIO(const std::vector<hsize_t>& datasetDims,
                                  const std::vector<hsize_t>& chunkDims,
                                  const std::vector<hsize_t>& selectionCount,
                                  hsize_t elementSize) {
    IOOpEstimate est{};
    int N = (int)datasetDims.size();
    hsize_t totalElements = product(selectionCount);
    est.totalBytes = totalElements * elementSize;

    if (!chunkDims.empty()) {
        // Chunked dataset: one I/O per chunk touched along each axis.
        // Ceiling division: a partially-covered chunk still costs one transaction.
        hsize_t chunkTransactions = 1;
        for (int i = 0; i < N; i++) {
            // this is just best calculation of ceil(selectionCount[i]/chunkDims[i]):
            hsize_t chunksTouched = (selectionCount[i] + chunkDims[i] - 1) / chunkDims[i];
            chunkTransactions *= std::max((hsize_t)1, chunksTouched);
        }
        est.transactions = chunkTransactions;
        est.bytesPerTransaction = est.transactions ? est.totalBytes / est.transactions : 0;
        return est;
    }

    // CODE FOR CONTINOUS CASE:
    // this code estimates how many blocks (transactions) of how many continues elements (runElements) are written

    // Contiguous dataset. The innermost axis's selected range is ALWAYS one
    // contiguous run on disk, whether or not it covers the full extent of
    // that axis. An outer axis i can be merged into that same run (so that
    // advancing i doesn't start a new transaction) only if every axis faster
    // than i (i.e. i+1 .. N-1) is fully covered -- otherwise consecutive
    // values of axis i land on disk with a gap between them.
    int splitAxis = N - 1; // innermost axis always starts the run
    for (int i = N - 2; i >= 0; i--) {
        if (selectionCount[i + 1] == datasetDims[i + 1]) {
            splitAxis = i; // axis i+1 was fully covered -> axis i merges in too
        } else {
            break;
        }
    }
    hsize_t transactions = 1; // number of separate write transations
    for (int i = 0; i < splitAxis; i++) {
        transactions *= selectionCount[i];
    }
    hsize_t runElements = 1; // number of continous elements in a transation
    for (int i = splitAxis; i < N; i++) {
        runElements *= selectionCount[i];
    }
    est.transactions = transactions;
    est.bytesPerTransaction = runElements * elementSize;
    return est;
}

IOOpEstimate combineEstimates(const IOOpEstimate& a, const IOOpEstimate& b) {
    IOOpEstimate c;
    c.transactions = a.transactions + b.transactions;
    c.totalBytes = a.totalBytes + b.totalBytes;
    c.bytesPerTransaction = c.transactions ? c.totalBytes / c.transactions : 0;
    return c;
}

IOOpEstimate repeatEstimate(const IOOpEstimate& e, hsize_t times) {
    return { e.transactions * times, e.bytesPerTransaction, e.totalBytes * times };
}

IOCostBreakdown estimateSlowConverterIO(hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                                         hsize_t numBins,
                                         const IOCostModel& readModel,
                                         const IOCostModel& writeModel) {
    IOCostBreakdown result;

    std::vector<hsize_t> standardDims = {stokes, depth, height, width};
    std::vector<hsize_t> chunkDims    = {1, 1, TILE_SIZE, TILE_SIZE};

    // ---------- 1st pass: SlowConverter.cc lines ~71-164 ----------

    // readFitsData(...) once per channel: not a hyperslab call (cfitsio),
    // one contiguous image plane per call -- uniform size, single add() is fine.
    {
        PhaseAccumulator acc;
        hsize_t bytesPerOp = height * width * sizeof(float);
        acc.add({ stokes * depth, bytesPerOp, stokes * depth * bytesPerOp }, readModel);
        result.phases.push_back(acc.toPhase("1st pass: FITS channel read (readFitsData)"));
    }

    // writeHdf5Data(standardDataSet, ...) once per channel — chunked, so this
    // decomposes into one write per {TILE_SIZE x TILE_SIZE} chunk. Boundary
    // chunks (where height/width isn't an exact multiple of TILE_SIZE) are
    // smaller than interior chunks; estimateHyperslabIO averages this single
    // call's chunks together, so predictSeconds sees one averaged size here.
    // If height/width are far from a TILE_SIZE multiple and you need this
    // exact, loop per-tile the way the 3rd pass already does below.
    {
        PhaseAccumulator acc;
        auto perChannel = estimateHyperslabIO(standardDims, chunkDims,
                                               {1, 1, height, width}, sizeof(float));
        acc.add(repeatEstimate(perChannel, stokes * depth), writeModel);
        result.phases.push_back(acc.toPhase("1st pass: standardDataSet write"));
    }

    // mipMaps.write(s, c) once per channel, per level. Each level is a
    // different size (double-precision buffer, shrinking dims) -- cost each
    // level separately, then sum.
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

    // ---------- 2nd pass: SlowConverter.cc lines ~205-289 ----------

    {
        PhaseAccumulator acc;
        hsize_t bytesPerOp = height * width * sizeof(float);
        acc.add({ stokes * depth, bytesPerOp, stokes * depth * bytesPerOp }, readModel);
        result.phases.push_back(acc.toPhase("2nd pass: FITS channel read (readFitsData)"));
    }

    // statsXY.write(...) once per stokes: MIN/MAX(float,4B), SUM/SUMSQ(double,8B),
    // NAN_COUNT(int64,8B), plus HISTOGRAM(int64,8B) if numBins > 0. Different
    // element sizes -> cost each sub-dataset separately.
    {
        PhaseAccumulator acc;
        hsize_t elemSizes[] = {4, 4, 8, 8, 8}; // MIN, MAX, SUM, SUMSQ, NAN_COUNT
        for (auto es : elemSizes) {
            auto e = estimateHyperslabIO({stokes, depth}, {}, {1, depth}, es);
            acc.add(repeatEstimate(e, stokes), writeModel);
        }
        if (numBins > 0) {
            auto h = estimateHyperslabIO({stokes, depth, numBins}, {}, {1, depth, numBins}, 8);
            acc.add(repeatEstimate(h, stokes), writeModel);
        }
        result.phases.push_back(acc.toPhase("2nd pass: statsXY write"));
    }

    if (depth > 1) {
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
        result.phases.push_back(acc.toPhase("2nd pass: statsXYZ write"));
    }

    // ---------- 3rd pass: tiled rotation, SlowConverter.cc lines ~300-436 ----------
    if (depth > 1) {
        std::vector<hsize_t> swizzledDims = {stokes, width, height, depth};
        std::vector<hsize_t> statsZDims   = {stokes, height, width};

        PhaseAccumulator readAcc, writeAcc, statsZAcc;

        // Mirrors the real tile grid loop exactly, so boundary tiles (smaller
        // than TILE_SIZE) get their own, correctly-sized cost lookup instead
        // of being averaged in with interior tiles.
        for (hsize_t xOffset = 0; xOffset < width; xOffset += TILE_SIZE) {
            hsize_t xSize = std::min(TILE_SIZE, width - xOffset);
            for (hsize_t yOffset = 0; yOffset < height; yOffset += TILE_SIZE) {
                hsize_t ySize = std::min(TILE_SIZE, height - yOffset);

                auto r = estimateHyperslabIO(standardDims, chunkDims,
                                              {1, depth, ySize, xSize}, sizeof(float));
                readAcc.add(repeatEstimate(r, stokes), readModel);

                auto w = estimateHyperslabIO(swizzledDims, {},
                                              {1, xSize, ySize, depth}, sizeof(float));
                writeAcc.add(repeatEstimate(w, stokes), writeModel);

                hsize_t elemSizes[] = {4, 4, 8, 8, 8};
                for (auto es : elemSizes) {
                    auto s = estimateHyperslabIO(statsZDims, {}, {1, ySize, xSize}, es);
                    statsZAcc.add(repeatEstimate(s, stokes), writeModel);
                }
            }
        }

        result.phases.push_back(readAcc.toPhase("3rd pass: standardDataSet tile read"));
        result.phases.push_back(writeAcc.toPhase("3rd pass: swizzledDataSet tile write"));
        result.phases.push_back(statsZAcc.toPhase("3rd pass: statsZ tile write"));
    }

    return result;
}

bool get_io_cost_model(const char* system_name, IOCostModel& setonixWrite, IOCostModel& setonixRead )
{
//   IOCostModel setonixWrite;
   setonixWrite.bandwidthCurve = {
       {4*1024,     20e6},   // fio --bs=4k  --rw=randwrite
       {64*1024,   150e6},   // fio --bs=64k --rw=randwrite
       {1024*1024, 600e6},   // fio --bs=1m  --rw=write
       {16*1024*1024, 1100e6} // fio --bs=16m --rw=write
   };    
       
//   IOCostModel setonixWrite;
   setonixRead.bandwidthCurve = {
       {4*1024,     20e6},   // fio --bs=4k  --rw=randwrite
       {64*1024,   150e6},   // fio --bs=64k --rw=randwrite
       {1024*1024, 600e6},   // fio --bs=1m  --rw=write
       {16*1024*1024, 1100e6} // fio --bs=16m --rw=write
   };

    
   // TODO: if (strcasecmp(system_name,"SETONIX") == 0 ){ return true; }  
    
   return true;
}