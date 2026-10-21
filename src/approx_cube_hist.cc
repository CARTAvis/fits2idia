/* approx_cube_hist : time the approximate cube (XYZ) histogram as CARTA would compute it
   from the per-channel histograms already stored in a fits2idia HDF5 file.

   Algorithm: SmartFastConverter::calcApproxCubeHistogram() (fits2idia), unchanged apart from
              - the per-bin DEBUG printf removed (it would dominate the timing),
              - the fractional counts accumulated as double (the original lambda took float).
   Optional  : -e also computes the EXACT cube histogram from 0/DATA, the way
              SmartFastConverter::doSecondPass() does it, for timing and accuracy comparison.

   HDF5 schema used (written by Converter::convert() / Stats::createDatasets()):
     0/DATA                        float  {[stokes,] [depth,] height, width}
     0/Statistics/XY/MIN, MAX      float  {[stokes,] depth}            (per channel)
     0/Statistics/XY/HISTOGRAM     int64  {[stokes,] depth, numBins}   (per channel)
     0/Statistics/XYZ/MIN, MAX     float  {[stokes]}                   (per cube)
     0/Statistics/XYZ/HISTOGRAM    int64  {[stokes,] numBins}          (per cube, reference)

   Build:  h5c++ -O3 -fopenmp -std=c++14 approx_cube_hist.cc -o approx_cube_hist
      or:  g++ -O3 -fopenmp -std=c++14 approx_cube_hist.cc -o approx_cube_hist -lhdf5_cpp -lhdf5
   Usage:  approx_cube_hist [options] FILE.hdf5   (run with -h for options)
*/

#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------------------------------------
// HDF5 helpers
// ------------------------------------------------------------------------------------------------
template <typename T> const H5::PredType& nativeType();
template <> const H5::PredType& nativeType<float>()   { return H5::PredType::NATIVE_FLOAT; }
template <> const H5::PredType& nativeType<int64_t>() { return H5::PredType::NATIVE_INT64; }

static std::vector<hsize_t> getDims(const H5::DataSet& ds) {
    H5::DataSpace sp = ds.getSpace();
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    if (rank > 0) sp.getSimpleExtentDims(dims.data());
    return dims;
}

// Reads the hyperslab (start, count) of ds into buf. A rank-0 (scalar) dataset is read whole.
// start/count must have the dataset's rank; leading axes beyond it are trimmed (like trimAxes()).
template <typename T>
static void readSlab(H5::DataSet& ds, T* buf, std::vector<hsize_t> start, std::vector<hsize_t> count) {
    H5::DataSpace fs = ds.getSpace();
    const size_t rank = (size_t)fs.getSimpleExtentNdims();
    if (rank == 0) {
        ds.read(buf, nativeType<T>());
        return;
    }
    while (start.size() > rank) start.erase(start.begin());
    while (count.size() > rank) count.erase(count.begin());
    fs.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    H5::DataSpace ms((int)count.size(), count.data());
    ds.read(buf, nativeType<T>(), ms, fs);
}

static bool linkExists(H5::H5Location& loc, const std::string& path) {
    // check each path component, H5Lexists fails on missing intermediate groups
    std::string partial;
    size_t pos = 0;
    while (true) {
        size_t next = path.find('/', pos);
        partial = path.substr(0, next);
        if (H5Lexists(loc.getId(), partial.c_str(), H5P_DEFAULT) <= 0) return false;
        if (next == std::string::npos) return true;
        pos = next + 1;
    }
}

// ------------------------------------------------------------------------------------------------
// Approximate cube histogram -- SmartFastConverter::calcApproxCubeHistogram(), inner loops.
// Accumulates nChannels channel histograms (row-major {nChannels, numBins}) into histXYZ.
// Can be called block by block; histXYZ keeps accumulating.
// ------------------------------------------------------------------------------------------------
static void accumulateApproxCubeHistogram(const int64_t* chanHists, const float* chanMinArr, const float* chanMaxArr,
                                          hsize_t nChannels, hsize_t numBins, double cubeMin, double cubeBinWidth,
                                          double* histogram_XYZ, int64_t& totalBinCount) {
    auto addFractionalCount = [&](int binIndex, double val) { histogram_XYZ[binIndex] += val; };

    for (hsize_t c = 0; c < nChannels; ++c) {
        double chanMin = chanMinArr[c];
        double chanMax = chanMaxArr[c];
        double chanRange = chanMax - chanMin;

        if (!std::isfinite(chanMin) || !std::isfinite(chanMax) || chanRange <= 0) {
            continue;
        }

        // width of a single bin of this channel's histogram
        double chanBinWidth = chanRange / numBins;
        const int64_t* hist = chanHists + c * numBins;

        for (size_t b = 0; b < numBins; ++b) {
            uint64_t binCount = hist[b];
            totalBinCount += binCount;

            if (binCount > 0) {
                // exact boundaries of the source (channel) bin
                double srcStart = chanMin + (b * chanBinWidth);
                double srcEnd = srcStart + chanBinWidth;

                // map them to floating-point bin indices of the cube histogram
                double destStartFloat = (srcStart - cubeMin) / cubeBinWidth;
                double destEndFloat = (srcEnd - cubeMin) / cubeBinWidth;

                int firstDestBin = std::max(0, (int)std::floor(destStartFloat));
                int lastDestBin = std::min((int)numBins - 1, (int)std::floor(destEndFloat));

                if (firstDestBin == lastDestBin) {
                    // source bin falls entirely inside one destination bin
                    addFractionalCount(firstDestBin, (double)binCount);
                } else {
                    // source bin spans two or more destination bins
                    double firstBinUpperBoundary = cubeMin + (firstDestBin + 1) * cubeBinWidth;
                    double firstBinFraction = (firstBinUpperBoundary - srcStart) / chanBinWidth;
                    addFractionalCount(firstDestBin, binCount * firstBinFraction);

                    double lastBinLowerBoundary = cubeMin + lastDestBin * cubeBinWidth;
                    double lastBinFraction = (srcEnd - lastBinLowerBoundary) / chanBinWidth;
                    addFractionalCount(lastDestBin, binCount * lastBinFraction);

                    // destination bins fully covered by the source bin
                    double middleBinFraction = cubeBinWidth / chanBinWidth;
                    for (int middleBin = firstDestBin + 1; middleBin < lastDestBin; ++middleBin) {
                        addFractionalCount(middleBin, binCount * middleBinFraction);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Exact cube histogram for a block of channels -- SmartFastConverter::doSecondPass(), inner loop.
// Channel-parallel, each channel fills its own partial histogram (no races), consolidated per block.
// ------------------------------------------------------------------------------------------------
static void accumulateExactCubeHistogram(const float* cube, hsize_t nChannels, hsize_t height, hsize_t width,
                                         const float* chanMinArr, const float* chanMaxArr, hsize_t numBins,
                                         double cubeMin, double cubeRange, int64_t* partial, int64_t* histXYZ) {
    std::fill(partial, partial + nChannels * numBins, (int64_t)0);

#pragma omp parallel for schedule(dynamic)
    for (hsize_t c = 0; c < nChannels; c++) {
        double chanMin = chanMinArr[c];
        double chanMax = chanMaxArr[c];
        double chanRange = chanMax - chanMin;
        bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);
        if (!chanHist) {
            continue;
        }
        int64_t* hist = partial + c * numBins;
        const float* chan = cube + c * height * width;
        for (hsize_t p = 0; p < height * width; p++) {
            float val = chan[p];
            if (std::isfinite(val)) {
                // Stats::accumulatePartialHistogram()
                hsize_t binIndex = std::min(numBins - 1, (hsize_t)(numBins * (val - cubeMin) / cubeRange));
                hist[binIndex]++;
            }
        }
    }

    // Stats::consolidateAndClearPartialHistogram(0)
    for (hsize_t c = 0; c < nChannels; c++) {
        for (hsize_t b = 0; b < numBins; b++) {
            histXYZ[b] += partial[c * numBins + b];
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Comparison of two cube histograms
// ------------------------------------------------------------------------------------------------
static void compareHistograms(const char* label, const std::vector<int64_t>& test, const std::vector<int64_t>& ref) {
    int64_t sumTest = 0, sumRef = 0, maxAbs = 0;
    double l1 = 0;
    size_t maxBin = 0;
    for (size_t b = 0; b < ref.size(); b++) {
        sumTest += test[b];
        sumRef += ref[b];
        int64_t d = std::llabs(test[b] - ref[b]);
        l1 += (double)d;
        if (d > maxAbs) { maxAbs = d; maxBin = b; }
    }
    printf("  %-34s total counts %lld vs %lld, sum|diff| = %.0f (%.4f%% of counts), max|diff| = %lld (bin %zu, ref %lld)\n",
           label, (long long)sumTest, (long long)sumRef, l1, sumRef ? 100.0 * l1 / (double)sumRef : 0.0,
           (long long)maxAbs, maxBin, (long long)ref[maxBin]);
}

// ------------------------------------------------------------------------------------------------
struct Options {
    std::string fileName;
    int repeats = 1;
    hsize_t histBlockChannels = 0;    // 0 = all channels in one read
    hsize_t dataBlockMB = 512;        // -e: max DATA block per read
    bool exact = false;
    long stokesSel = -1;              // -1 = all
    std::string dumpFile;
};

static void usage(const char* prog) {
    printf("Usage: %s [options] FILE.hdf5\n"
           "Times reading the per-channel histograms of a fits2idia HDF5 file and computing the approximate\n"
           "cube (XYZ) histogram from them (SmartFastConverter::calcApproxCubeHistogram).\n\n"
           "Options:\n"
           "  -n N    repeat the whole measurement N times (default 1); repeats after the first read from page cache\n"
           "  -B N    read the channel histograms in blocks of N channels (default: all channels in one read)\n"
           "  -s S    process only Stokes S (default: all)\n"
           "  -e      also compute the EXACT cube histogram from 0/DATA (doSecondPass algorithm) and time it\n"
           "  -M MB   with -e: max size of one DATA read in MB (default 512)\n"
           "  -o F    write the cube histogram(s) of the last Stokes processed to text file F\n"
           "  -h      this help\n",
           prog);
}

static bool parseArgs(int argc, char** argv, Options& o) {
    int opt;
    while ((opt = getopt(argc, argv, "n:B:s:eM:o:h")) != -1) {
        switch (opt) {
            case 'n': o.repeats = std::max(1, atoi(optarg)); break;
            case 'B': o.histBlockChannels = strtoull(optarg, nullptr, 10); break;
            case 's': o.stokesSel = atol(optarg); break;
            case 'e': o.exact = true; break;
            case 'M': o.dataBlockMB = std::max((hsize_t)1, (hsize_t)strtoull(optarg, nullptr, 10)); break;
            case 'o': o.dumpFile = optarg; break;
            case 'h': default: usage(argv[0]); return false;
        }
    }
    if (optind != argc - 1) {
        usage(argv[0]);
        return false;
    }
    o.fileName = argv[optind];
    return true;
}

// Timings of one full run (summed over the Stokes processed)
struct RunTimes {
    double open_ms = 0, readMinMax_ms = 0, readHist_ms = 0, cubeMinMax_ms = 0, approx_ms = 0;
    double readData_ms = 0, exact_ms = 0;
    hsize_t minMaxBytes = 0, histBytes = 0, dataBytes = 0;
    hsize_t histReads = 0, dataReads = 0;
};

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    H5::Exception::dontPrint();

    std::vector<RunTimes> runs;
    try {
        for (int rep = 0; rep < opt.repeats; rep++) {
            RunTimes t;
            const bool report = (rep == opt.repeats - 1); // accuracy report / dump only once

            auto t0 = Clock::now();
            H5::H5File file(opt.fileName, H5F_ACC_RDONLY);
            H5::Group group = file.openGroup("0");
            H5::DataSet xyMin = group.openDataSet("Statistics/XY/MIN");
            H5::DataSet xyMax = group.openDataSet("Statistics/XY/MAX");
            H5::DataSet xyHist = group.openDataSet("Statistics/XY/HISTOGRAM");
            t.open_ms = msSince(t0);

            // dimensions: XY HISTOGRAM is {[stokes,] depth, numBins}
            auto hd = getDims(xyHist);
            if (hd.size() < 2) {
                std::cerr << "Statistics/XY/HISTOGRAM has rank " << hd.size()
                          << " -> 2D image (single channel), no cube histogram to compute" << std::endl;
                return 0;
            }
            const hsize_t numBins = hd.back();
            const hsize_t depth = hd[hd.size() - 2];
            const hsize_t stokes = hd.size() == 3 ? hd[0] : 1;

            if (rep == 0) {
                printf("File            : %s\n", opt.fileName.c_str());
                printf("Stokes x depth  : %llu x %llu, numBins = %llu\n", (unsigned long long)stokes,
                       (unsigned long long)depth, (unsigned long long)numBins);
                printf("XY HISTOGRAM    : %.3f MB per Stokes (%llu channels x %llu bins x 8 B)\n",
                       depth * numBins * 8 / 1e6, (unsigned long long)depth, (unsigned long long)numBins);
                if (group.attrExists("HDF5_CONVERTER_TYPE")) {
                    H5::Attribute a = group.openAttribute("HDF5_CONVERTER_TYPE");
                    std::string v;
                    a.read(a.getStrType(), v);
                    printf("Converter type  : %s\n", v.c_str());
                }
                if (group.attrExists("HDF5_PARAM_CUBE_HISTOGRAM_APPROXIMATE")) {
                    hbool_t v = 0;
                    group.openAttribute("HDF5_PARAM_CUBE_HISTOGRAM_APPROXIMATE").read(H5::PredType::NATIVE_HBOOL, &v);
                    printf("Stored XYZ hist : %s\n", v ? "APPROXIMATE (flag set by converter)" : "exact");
                }
#ifdef _OPENMP
                printf("OpenMP threads  : %d (used by -e only)\n", omp_get_max_threads());
#endif
                printf("\n");
            }
            if (depth <= 1) {
                std::cerr << "depth = 1 -> no cube histogram to compute (it equals the channel histogram)" << std::endl;
                return 0;
            }

            const hsize_t histBlock = (opt.histBlockChannels == 0) ? depth : std::min(opt.histBlockChannels, depth);

            // optional datasets
            const bool haveRef = linkExists(group, "Statistics/XYZ/HISTOGRAM");
            H5::DataSet xyzHist;
            if (haveRef) xyzHist = group.openDataSet("Statistics/XYZ/HISTOGRAM");

            H5::DataSet dataSet;
            hsize_t height = 0, width = 0, dataBlock = 0;
            if (opt.exact) {
                dataSet = group.openDataSet("DATA");
                auto dd = getDims(dataSet);
                height = dd[dd.size() - 2];
                width = dd[dd.size() - 1];
                dataBlock = std::max((hsize_t)1, std::min(depth, (hsize_t)((opt.dataBlockMB * 1000000ULL) / (height * width * sizeof(float)))));
            }

            std::vector<float> chanMin(depth), chanMax(depth);
            std::vector<int64_t> histBuf(histBlock * numBins);
            std::vector<double> histXYZ_d(numBins);
            std::vector<int64_t> approx(numBins), exact, ref;
            std::vector<float> dataBuf;
            std::vector<int64_t> partial;
            if (opt.exact) {
                dataBuf.resize(dataBlock * height * width);
                partial.resize(dataBlock * numBins);
                exact.resize(numBins);
            }

            const hsize_t s0 = opt.stokesSel >= 0 ? (hsize_t)opt.stokesSel : 0;
            const hsize_t s1 = opt.stokesSel >= 0 ? (hsize_t)opt.stokesSel + 1 : stokes;
            if (s0 >= stokes) {
                std::cerr << "Stokes " << opt.stokesSel << " out of range (file has " << stokes << ")" << std::endl;
                return 1;
            }

            double lastCubeMin = 0, lastBinWidth = 0;
            for (hsize_t s = s0; s < s1; s++) {
                // ---- I/O 1: per-channel MIN / MAX
                t0 = Clock::now();
                readSlab(xyMin, chanMin.data(), {s, 0}, {1, depth});
                readSlab(xyMax, chanMax.data(), {s, 0}, {1, depth});
                t.readMinMax_ms += msSince(t0);
                t.minMaxBytes += 2 * depth * sizeof(float);

                // ---- cube min/max from channel min/max (Stats::accumulateStatsToCounter)
                t0 = Clock::now();
                float cubeMinF = std::numeric_limits<float>::max(), cubeMaxF = -std::numeric_limits<float>::max();
                bool anyFinite = false;
                for (hsize_t c = 0; c < depth; c++) {
                    if (std::isfinite(chanMax[c])) {
                        cubeMinF = std::fmin(cubeMinF, chanMin[c]);
                        cubeMaxF = std::fmax(cubeMaxF, chanMax[c]);
                        anyFinite = true;
                    }
                }
                double cubeMin = anyFinite ? cubeMinF : NAN;
                double cubeMax = anyFinite ? cubeMaxF : NAN;
                double cubeRange = cubeMax - cubeMin;
                bool cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
                double cubeBinWidth = cubeRange / numBins;
                t.cubeMinMax_ms += msSince(t0);

                if (!cubeHist) {
                    printf("Stokes %llu: cube range not finite/positive -> no cube histogram\n", (unsigned long long)s);
                    continue;
                }
                lastCubeMin = cubeMin;
                lastBinWidth = cubeBinWidth;

                // ---- I/O 2 + calculation: channel histograms, block by block
                std::fill(histXYZ_d.begin(), histXYZ_d.end(), 0.0);
                int64_t totalBinCount = 0;
                for (hsize_t c0 = 0; c0 < depth; c0 += histBlock) {
                    const hsize_t nCh = std::min(histBlock, depth - c0);

                    t0 = Clock::now();
                    readSlab(xyHist, histBuf.data(), {s, c0, 0}, {1, nCh, numBins});
                    t.readHist_ms += msSince(t0);
                    t.histBytes += nCh * numBins * sizeof(int64_t);
                    t.histReads++;

                    t0 = Clock::now();
                    accumulateApproxCubeHistogram(histBuf.data(), chanMin.data() + c0, chanMax.data() + c0, nCh, numBins,
                                                  cubeMin, cubeBinWidth, histXYZ_d.data(), totalBinCount);
                    t.approx_ms += msSince(t0);
                }
                // final rounding to integer counts (as in calcApproxCubeHistogram)
                t0 = Clock::now();
                for (hsize_t b = 0; b < numBins; b++) {
                    approx[b] = (int64_t)std::round(histXYZ_d[b]);
                }
                t.approx_ms += msSince(t0);

                // ---- optional: exact cube histogram from DATA (doSecondPass)
                if (opt.exact) {
                    std::fill(exact.begin(), exact.end(), (int64_t)0);
                    for (hsize_t c0 = 0; c0 < depth; c0 += dataBlock) {
                        const hsize_t nCh = std::min(dataBlock, depth - c0);
                        t0 = Clock::now();
                        readSlab(dataSet, dataBuf.data(), {s, c0, 0, 0}, {1, nCh, height, width});
                        t.readData_ms += msSince(t0);
                        t.dataBytes += nCh * height * width * sizeof(float);
                        t.dataReads++;

                        t0 = Clock::now();
                        accumulateExactCubeHistogram(dataBuf.data(), nCh, height, width, chanMin.data() + c0,
                                                     chanMax.data() + c0, numBins, cubeMin, cubeRange, partial.data(),
                                                     exact.data());
                        t.exact_ms += msSince(t0);
                    }
                }

                // ---- accuracy report (not timed)
                if (report) {
                    printf("Stokes %llu: cube min/max = %.8g / %.8g, channel-histogram counts = %lld\n",
                           (unsigned long long)s, cubeMin, cubeMax, (long long)totalBinCount);
                    if (haveRef) {
                        ref.assign(numBins, 0);
                        readSlab(xyzHist, ref.data(), {s, 0}, {1, numBins});
                        compareHistograms("approx vs stored XYZ/HISTOGRAM:", approx, ref);
                        if (opt.exact) compareHistograms("exact  vs stored XYZ/HISTOGRAM:", exact, ref);
                    }
                    if (opt.exact) compareHistograms("approx vs exact (from DATA):", approx, exact);
                }
            } // stokes

            if (report && !opt.dumpFile.empty()) {
                std::ofstream out(opt.dumpFile);
                out << "# bin bin_start bin_end approx" << (opt.exact ? " exact" : "") << (haveRef ? " stored" : "") << "\n";
                for (hsize_t b = 0; b < numBins; b++) {
                    out << b << " " << lastCubeMin + b * lastBinWidth << " " << lastCubeMin + (b + 1) * lastBinWidth << " "
                        << approx[b];
                    if (opt.exact) out << " " << exact[b];
                    if (haveRef) out << " " << ref[b];
                    out << "\n";
                }
                printf("Histogram(s) of the last Stokes written to %s\n", opt.dumpFile.c_str());
            }

            runs.push_back(t);
        } // repeats
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getFuncName() << " : " << e.getDetailMsg() << std::endl;
        return 1;
    }

    // ---- timing report
    auto mbps = [](hsize_t bytes, double ms) { return ms > 0 ? bytes / 1e6 / (ms / 1e3) : 0.0; };
    printf("\nTIMING (ms, summed over the Stokes processed)\n");
    printf("%-4s %9s %11s %11s %11s %11s %12s", "run", "open", "read min/max", "read hist", "cube minmax", "approx calc", "approx total");
    if (opt.exact) printf(" %11s %11s %12s", "read DATA", "exact calc", "exact total");
    printf("\n");
    for (size_t r = 0; r < runs.size(); r++) {
        const auto& t = runs[r];
        double approxTotal = t.readMinMax_ms + t.readHist_ms + t.cubeMinMax_ms + t.approx_ms;
        printf("%-4zu %9.3f %11.3f %11.3f %11.3f %11.3f %12.3f", r + 1, t.open_ms, t.readMinMax_ms, t.readHist_ms,
               t.cubeMinMax_ms, t.approx_ms, approxTotal);
        if (opt.exact) {
            double exactTotal = t.readMinMax_ms + t.cubeMinMax_ms + t.readData_ms + t.exact_ms;
            printf(" %11.3f %11.3f %12.3f", t.readData_ms, t.exact_ms, exactTotal);
        }
        printf("\n");
    }
    const auto& f = runs.front();
    printf("\nPer run: channel histograms %.3f MB in %llu read(s) -> %.1f MB/s on the first run",
           f.histBytes / 1e6, (unsigned long long)f.histReads, mbps(f.histBytes, f.readHist_ms));
    if (runs.size() > 1) {
        double best = runs[1].readHist_ms;
        for (size_t r = 1; r < runs.size(); r++) best = std::min(best, runs[r].readHist_ms);
        printf(", %.1f MB/s best of the repeats (page cache)", mbps(f.histBytes, best));
    }
    printf("\n");
    if (opt.exact) {
        printf("         DATA %.3f MB in %llu read(s) -> %.1f MB/s on the first run\n", f.dataBytes / 1e6,
               (unsigned long long)f.dataReads, mbps(f.dataBytes, f.readData_ms));
    }
    printf("NOTE: the first run is only a cold-cache read if the file is not in the OS page cache\n"
           "      (e.g. freshly copied, or after 'echo 3 > /proc/sys/vm/drop_caches' as root).\n");
    return 0;
}
