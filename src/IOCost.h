/* I/O transaction-count estimator for fits2idia benchmarking / cost modelling.
   Not part of the conversion path itself — used to predict how many discrete
   OS-level read()/write() transactions a given HDF5 hyperslab call will turn
   into, and how big each one is, so that runtime can be estimated from a
   per-machine throughput(block_size) curve rather than raw byte counts alone.
*/

#ifndef __IO_COST_H
#define __IO_COST_H

#include "common.h"

struct IOOpEstimate {
    hsize_t transactions;        // estimated discrete OS-level read()/write() calls
    hsize_t bytesPerTransaction; // approx size of each one (bytes)
    hsize_t totalBytes;
};

// datasetDims    : full on-disk dims of the dataset
// chunkDims      : chunk dims if chunked, or {} if contiguous
// selectionCount : the hyperslab "count" being read/written in this call
// elementSize    : sizeof(float) / sizeof(int64_t) / etc.
//
// Chunked:     one transaction per chunk touched (ceil(count[i]/chunk[i]) per axis, multiplied).
// Contiguous:  find the longest trailing run of axes where selectionCount[i] == datasetDims[i]
//              (fully covered -> stays one contiguous run on disk); every axis above that
//              boundary is a separate transaction.
IOOpEstimate estimateHyperslabIO(const std::vector<hsize_t>& datasetDims,
                                  const std::vector<hsize_t>& chunkDims,
                                  const std::vector<hsize_t>& selectionCount,
                                  hsize_t elementSize);

// Measured cost model: pass fio/dd results directly as (block size in bytes,
// achieved bandwidth in bytes/sec) points, sorted ascending by size. Looking
// up a size between two measured points uses log-log linear interpolation
// (appropriate here because block size and bandwidth both span orders of
// magnitude and the relationship curves, rather than being a straight line,
// on a plain linear scale). Sizes outside the measured range clamp to the
// nearest endpoint.
//
// NOTE (important difference from a fitted 2-parameter model): predictSeconds
// looks up bandwidth at THIS estimate's own bytesPerTransaction. That means
// you can no longer combine several differently-sized calls into one
// aggregate IOOpEstimate before calling predictSeconds and expect the same
// answer as calling it on each and summing -- the aggregate's "average" size
// would land on the wrong point of a non-linear curve. Keep differently-sized
// calls separate; use PhaseAccumulator below to do that correctly while still
// reporting one combined total per phase.
struct IOCostModel {
    std::vector<std::pair<hsize_t, double>> bandwidthCurve; // (sizeBytes, bytesPerSec), sorted by size

    double bandwidthAt(hsize_t sizeBytes) const {
        if (bandwidthCurve.empty()) return 1.0; // unconfigured guard, avoids div-by-zero
        if (sizeBytes <= bandwidthCurve.front().first) return bandwidthCurve.front().second;
        if (sizeBytes >= bandwidthCurve.back().first)  return bandwidthCurve.back().second;
        for (size_t i = 1; i < bandwidthCurve.size(); i++) {
            hsize_t size2 = bandwidthCurve[i].first;
            if (sizeBytes <= size2) {
                hsize_t size1 = bandwidthCurve[i - 1].first;
                double logS  = std::log((double)sizeBytes);
                double logS1 = std::log((double)size1);
                double logS2 = std::log((double)size2);
                double logBW1 = std::log(bandwidthCurve[i - 1].second);
                double logBW2 = std::log(bandwidthCurve[i].second);
                double t = (logS - logS1) / (logS2 - logS1);
                return std::exp(logBW1 + t * (logBW2 - logBW1));
            }
        }
        return bandwidthCurve.back().second;
    }

    // NOTE: only valid when est represents a SINGLE size (one call, or several
    // repeats of the exact same size -- see repeatEstimate). Do not call this
    // on an estimate produced by combining different sizes together.
    double predictSeconds(const IOOpEstimate& est) const {
        if (est.bytesPerTransaction == 0) return 0.0;
        return (double)est.totalBytes / bandwidthAt(est.bytesPerTransaction);
    }
};

// Combine two estimates that happen in the same "phase" (adds transactions
// and bytes; recomputes the average bytesPerTransaction for display only).
// Use only for REPORTING two same-size groups together -- never pass the
// result to IOCostModel::predictSeconds if the two inputs had different
// bytesPerTransaction (see PhaseAccumulator instead).
IOOpEstimate combineEstimates(const IOOpEstimate& a, const IOOpEstimate& b);

// Scale one estimate by a repeat count (e.g. "this call happens `depth`
// times"). Always safe to predictSeconds on the result -- it's still one size.
IOOpEstimate repeatEstimate(const IOOpEstimate& e, hsize_t times);

// One line of the cost breakdown table.
struct IOPhaseCost {
    std::string name;
    IOOpEstimate estimate; // totals, for display only
    double predictedSeconds;
};

// Accumulates a phase made of several differently-sized calls: each call is
// costed individually against the model (correct with a non-linear BW(size)
// curve), while transactions/bytes are summed only for the printed report.
struct PhaseAccumulator {
    hsize_t transactions = 0;
    hsize_t totalBytes = 0;
    double predictedSeconds = 0.0;

    void add(const IOOpEstimate& est, const IOCostModel& model) {
        transactions += est.transactions;
        totalBytes += est.totalBytes;
        predictedSeconds += model.predictSeconds(est);
    }

    IOPhaseCost toPhase(const std::string& name) const {
        hsize_t avgBytesPerOp = transactions ? totalBytes / transactions : 0;
        return { name, { transactions, avgBytesPerOp, totalBytes }, predictedSeconds };
    }
};

// Full breakdown for one converter run.
struct IOCostBreakdown {
    std::vector<IOPhaseCost> phases;

    double totalSeconds() const {
        double t = 0;
        for (auto& p : phases) t += p.predictedSeconds;
        return t;
    }

    void print(std::ostream& out = std::cout) const {
        out << std::endl;
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        out << "Estimated execution time due to I/O operations:" << std::endl;
        for (auto& p : phases) {
            out << p.name
                << " : transactions=" << p.estimate.transactions
                << ", avg bytes/op=" << p.estimate.bytesPerTransaction
                << ", total bytes=" << p.estimate.totalBytes
                << ", predicted=" << p.predictedSeconds << " s"
                << std::endl;
        }
        out << "TOTAL predicted I/O time: " << totalSeconds() << " s" << std::endl;
        out << "--------------------------------------------------------------------------" << std::endl;
        out << std::endl;
    }
};

void addTiledRotationPhases(IOCostBreakdown& result,
                             hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                             const IOCostModel& readModel, const IOCostModel& writeModel);

bool get_io_cost_model(const char* system_name, IOCostModel& setonixRead, IOCostModel& setonixWrite, bool use_random_read_write=false );

#endif
