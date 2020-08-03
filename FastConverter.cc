#include "Converter.h"

// TODO do we need these?
FastConverter::FastConverter(std::string inputFileName, std::string outputFileName, bool progress) :
Converter(inputFileName, outputFileName, progress) {}

void FastConverter::reportMemoryUsage() {
    std::unordered_map<std::string, hsize_t> sizes;

    sizes["Main dataset"] = depth * height * width * sizeof(float);
    sizes["Mipmaps"] = MipMaps::size(standardDims, {depth, height, width});
    sizes["XY stats"] = Stats::size({depth}, numBins);

    if (depth > 1) {
        sizes["Rotation"] = sizes["Main dataset"];
        sizes["XYZ stats"] = Stats::size({}, numBins, depth);
        sizes["Z stats"] = Stats::size({height, width});
    }

    hsize_t total(0);

    std::cout << "APPROXIMATE MEMORY REQUIREMENTS:" << std::endl;

    for (auto& kv : sizes) {
        std::cout << kv.first << ":\t" << kv.second * 1e-9 << " GB" << std::endl;
        total += kv.second;
    }

    std::string note = "";
    if (depth > 1) {
        total -= std::min(sizes["Mipmaps"], sizes["Rotation"]);
        note = " (Rotated dataset and mipmaps are not allocated at the same time.)";
    }

    std::cout << "TOTAL:\t" << total * 1e-9 << "GB" << note << std::endl;
}

void FastConverter::copyAndCalculate() {
    const hsize_t pixelProgressStride = std::max((hsize_t)1, (hsize_t)(width * height / 100));
    const hsize_t channelProgressStride = std::max((hsize_t)1, (hsize_t)(depth / 100));

    TIMER(timer.start("Allocate"););

    // Process one stokes at a time
    hsize_t cubeSize = depth * height * width;
    standardCube = new float[cubeSize];

    statsXY.createBuffers({depth});

    if (depth > 1) {
        statsXYZ.createBuffers({}, depth);
        statsZ.createBuffers({height, width});
    }

    mipMaps.createBuffers({depth, height, width});

    std::string timerLabelXYRotation = depth > 1 ? "XY statistics and rotation" : "XY statistics";

    for (unsigned int currentStokes = 0; currentStokes < stokes; currentStokes++) {
        DEBUG(std::cout << "Processing Stokes " << currentStokes << "..." << std::endl;);
        PROGRESS("Stokes " << currentStokes << ":" << std::endl);

        // Read data into memory space
        TIMER(timer.start("Read"););
        DEBUG(std::cout << "+ Reading main dataset..." << std::flush;);
        readFitsData(inputFilePtr, 0, currentStokes, cubeSize, standardCube);

        // We have to allocate the swizzled cube for each stokes because we free it to make room for mipmaps
        if (depth > 1) {
            TIMER(timer.start("Allocate"););
            rotatedCube = new float[cubeSize];
        }

        DEBUG(std::cout << " " << timerLabelXYRotation <<  "..." << std::flush;);
        PROGRESS("\tMain loop\t");
        TIMER(timer.start(timerLabelXYRotation););

        // First loop calculates stats for each XY slice and rotates the dataset

#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");
            StatsCounter counterXY;

            auto& indexXY = i;
            std::function<void(float)> accumulate;

            auto lazy_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazy(val);
            };

            auto first_accumulate = [&] (float val) {
                counterXY.accumulateFiniteLazyFirst(val);
                accumulate = lazy_accumulate;
            };

            accumulate = first_accumulate;

            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    auto sourceIndex = k + width * j + (height * width) * i;
                    auto destIndex = i + depth * j + (height * depth) * k;
                    auto& val = standardCube[sourceIndex];

                    if (depth > 1) {
                        rotatedCube[destIndex] = val;
                    }

                    // Accumulate XY stats
                    if (std::isfinite(val)) {
                        accumulate(val);
                    } else {
                        counterXY.accumulateNonFinite();
                    }
                }
            }

            // Final correction of XY min and max
            statsXY.copyStatsFromCounter(indexXY, height * width, counterXY);
        }

        PROGRESS(std::endl);

        if (depth > 1) {
            // Consolidate XY stats into XYZ stats
            DEBUG(std::cout << " XYZ statistics..." << std::flush;);
            PROGRESS("\tXYZ stats" << std::endl);
            TIMER(timer.start("XYZ statistics"););

            StatsCounter counterXYZ;

            for (hsize_t i = 0; i < depth; i++) {
                auto& indexXY = i;
                statsXY.accumulateStatsToCounter(counterXYZ, indexXY);
            }

            statsXYZ.copyStatsFromCounter(0, depth * height * width, counterXYZ);

            // Second loop calculates stats for each Z profile (i.e. average/min/max XY slices)

            DEBUG(std::cout << " Z statistics... " << std::flush;);
            PROGRESS("\tZ stats\t\t");
            TIMER(timer.start("Z statistics"););

#pragma omp parallel for
            for (hsize_t j = 0; j < height; j++) {
                for (hsize_t k = 0; k < width; k++) {
                    StatsCounter counterZ;

                    auto indexZ = k + j * width;
                    PROGRESS_DECIMATED(indexZ, pixelProgressStride, ".");

                    for (hsize_t i = 0; i < depth; i++) {
                        auto sourceIndex = k + width * j + (height * width) * i;
                        auto& val = standardCube[sourceIndex];

                        if (std::isfinite(val)) {
                            // Not lazy; too much risk of encountering an ascending / descending sequence.
                            counterZ.accumulateFinite(val);
                        } else {
                            counterZ.accumulateNonFinite();
                        }
                    }

                    statsZ.copyStatsFromCounter(indexZ, depth, counterZ);
                }
            }

            PROGRESS(std::endl);
        }

        // Third loop handles histograms

        DEBUG(std::cout << " Histograms..." << std::flush;);
        PROGRESS("\tHistograms\t");
        TIMER(timer.start("Histograms"););

        double cubeMin;
        double cubeMax;
        double cubeRange;
        bool cubeHist(false);

        if (depth > 1) {
            cubeMin = statsXYZ.minVals[0];
            cubeMax = statsXYZ.maxVals[0];
            cubeRange = cubeMax - cubeMin;
            cubeHist = std::isfinite(cubeMin) && std::isfinite(cubeMax) && cubeRange > 0;
        }

        statsXY.clearHistogramBuffers();
        statsXYZ.clearHistogramBuffers();

#pragma omp parallel for
        for (hsize_t i = 0; i < depth; i++) {
            PROGRESS_DECIMATED(i, channelProgressStride, "|");

            auto& indexXY = i;
            double chanMin = statsXY.minVals[indexXY];
            double chanMax = statsXY.maxVals[indexXY];
            double chanRange = chanMax - chanMin;

            bool chanHist(std::isfinite(chanMin) && std::isfinite(chanMax) && chanRange > 0);

            if (!chanHist && !cubeHist) {
                continue; // skip the loop entirely
            }

            auto doChannelHistogram = [&] (float val) {
                // XY histogram
                statsXY.accumulateHistogram(val, chanMin, chanRange, i);
            };

            auto doCubeHistogram = [&] (float val) {
                // Partial XYZ histogram
                statsXYZ.accumulatePartialHistogram(val, cubeMin, cubeRange, i);
            };

            auto doNothing = [&] (float val) {
                UNUSED(val);
            };

            std::function<void(float)> channelHistogramFunc = doChannelHistogram;
            std::function<void(float)> cubeHistogramFunc = doCubeHistogram;

            if (!chanHist) {
                channelHistogramFunc = doNothing;
            }

            if (!cubeHist) {
                cubeHistogramFunc = doNothing;
            }

            for (hsize_t j = 0; j < width * height; j++) {
                auto& val = standardCube[i * width * height + j];

                if (std::isfinite(val)) {
                    channelHistogramFunc(val);
                    cubeHistogramFunc(val);
                }
            } // end of XY loop
        } // end of parallel Z loop

        if (depth > 1) {
            // Consolidate partial XYZ histograms into final histogram
            statsXYZ.consolidatePartialHistogram();
        }

        PROGRESS(std::endl);

        DEBUG(std::cout << " Writing main and rotated datasets... " << std::flush;);
        PROGRESS("\tWrite data" << std::endl);
        TIMER(timer.start("Write"););

        std::vector<hsize_t> memDims = {depth, height, width};
        std::vector<hsize_t> count = trimAxes({1, depth, height, width}, N);
        std::vector<hsize_t> start = trimAxes({currentStokes, 0, 0, 0}, N);
//         writeHdf5Data(standardDataSet, standardCube, memDims, count, start);
        H5outputfile.write_dataset_nd(standardDataSet,
            memDims, standardCube, count, start);

        if (depth > 1) {
            // This all technically worked if we reused the standard filespace and memspace
            // But it's probably not a good idea to rely on two incorrect values cancelling each other out
            std::vector<hsize_t> swizzledCount = trimAxes({1, width, height, depth}, N);
            std::vector<hsize_t> swizzledMemDims = {width, height, depth};
//             writeHdf5Data(swizzledDataSet, rotatedCube, swizzledMemDims, swizzledCount, start);
            H5outputfile.write_dataset_nd(swizzledDataSet,
                swizzledMemDims, rotatedCube, swizzledCount, start);
        }

        // After writing and before mipmaps, we free the swizzled memory. We allocate it again next Stokes.
        if (depth > 1) {
            DEBUG(std::cout << " Freeing memory from rotated dataset..." << std::flush;);
            TIMER(timer.start("Free"););

            delete[] rotatedCube;
        }

        // Fourth loop handles mipmaps

        // In the fast algorithm, we keep one Stokes of mipmaps in memory at once and parallelise by channel
        DEBUG(std::cout << " Mipmaps..." << std::endl;);
        PROGRESS("\tMipmaps\t\t");
        TIMER(timer.start("Mipmaps"););

#pragma omp parallel for
        for (hsize_t c = 0; c < depth; c++) {
            PROGRESS_DECIMATED(c, channelProgressStride, "|");
            for (hsize_t y = 0; y < height; y++) {
                for (hsize_t x = 0; x < width; x++) {
                    auto sourceIndex = x + width * y + (height * width) * c;
                    auto& val = standardCube[sourceIndex];
                    if (std::isfinite(val)) {
                        mipMaps.accumulate(val, x, y, c);
                    }
                }
            }
        } // end of mipmap loop
        PROGRESS(std::endl);

        // Final mipmap calculation
        mipMaps.calculate();

        TIMER(timer.start("Write"););
        PROGRESS("\tWrite stats & mipmaps" << std::endl);

        // Write the mipmaps
        mipMaps.write(H5outputfile, currentStokes, 0);

        // Write the statistics
        statsXY.write(H5outputfile,{1, depth}, {currentStokes, 0});

        if (depth > 1) {
            statsXYZ.write(H5outputfile, {1}, {currentStokes});
            statsZ.write(H5outputfile, {1, height, width}, {currentStokes, 0, 0});
        }

        // Clear the mipmaps before the next Stokes
        TIMER(timer.start("Mipmaps"););
        mipMaps.resetBuffers();

    } // end of Stokes loop

    // Free memory
    DEBUG(std::cout << "Freeing memory from main dataset... " << std::endl;);
    TIMER(timer.start("Free"););

    delete[] standardCube;
}
