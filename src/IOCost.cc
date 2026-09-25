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

bool get_io_cost_model(const char* system_name, IOCostModel& readBW, IOCostModel& writeBW, bool use_random_read_write /*=false*/ )
{
   // see 20260916_IO_BW_measurements_SETONIX.odt
   if (strcasecmp(system_name,"setonix") == 0 ) {
      printf("BW INFO : using BW measured on SETONIX (normal Luster partition)\n");

      if (use_random_read_write) {
         writeBW.bandwidthCurve = { // randwrite_bw_vs_size.cc
             // MORNING : /scratch less used: 20260916_IO_BW_measurements_SETONIX.odt
             /*{ 4*1024 , 1.597e6 },
             { 64*1024 , 15.9e6 },
             { 1024*1024 , 353e6 },
             { 16*1024*1024 , 1250e6 },
             { 32*1024*1024 , 1308e6 },
             { 64*1024*1024 , 2169e6 },
             { 128*1024*1024 , 610e6 } */                          

             // AFTERNOON : /scratch more busy (more users etc): 20260916_IO_BW_measurements_SETONIX-REPEAT-NON-SSD.odt
             { 4*1024 , 0.0239e6 },
             { 64*1024 , 0.858e6 },
             { 1024*1024 , 50.7e6 },
             { 16*1024*1024 , 156e6 },
             { 32*1024*1024 , 225e6 },
             { 64*1024*1024 , 290e6 },
             { 128*1024*1024 , 652e6 },
             { 256*1024*1024 , 953e6 },
             { 512*1024*1024 , 982e6 }
         };
      } else {
         writeBW.bandwidthCurve = { // write_bw_vs_size.cc
             // MORNING : /scratch less used: 20260916_IO_BW_measurements_SETONIX.odt
             /*{ 4*1024 , 3.773e6 },
             { 64*1024 , 38.2e6 },
             { 1024*1024 , 326e6 },
             { 16*1024*1024 , 1414e6 },
             { 32*1024*1024 , 1491e6 },
             { 64*1024*1024 , 1974e6 },
             { 128*1024*1024 , 2297e6 }*/

             // AFTERNOON : /scratch more busy (more users etc): 20260916_IO_BW_measurements_SETONIX-REPEAT-NON-SSD.odt
             { 4*1024 , 1.376e6 },
             { 64*1024 , 3.445e6 },
             { 1024*1024 , 57.5e6 },
             { 16*1024*1024 , 172e6 },
             { 32*1024*1024 , 280e6 },
             { 64*1024*1024 , 403e6 },
             { 128*1024*1024 , 525e6 },
             { 256*1024*1024 , 1194e6 },
             { 512*1024*1024 , 1350e6 }
         };
      }
      if (use_random_read_write) {
         readBW.bandwidthCurve = { // randread_bw_vs_size.cc
             { 4*1024 , 0.22e6 },
             { 64*1024 , 4.317e6 },
             { 1024*1024 , 24.0e6 },
             { 16*1024*1024 , 3230e6 },
             { 32*1024*1024 , 3771e6 },
             { 64*1024*1024 , 4658e6 },
             { 128*1024*1024 , 5369e6 },
             { 256*1024*1024 , 5863e6 },
             { 512*1024*1024 , 5831e6 }
         };
      } else {
         readBW.bandwidthCurve = { // read_bw_vs_size.cc
             { 4*1024 , 8.237e6 },
             { 64*1024 , 22.2e6 },
             { 1024*1024 , 170e6 },
             { 16*1024*1024 , 3379e6 },
             { 32*1024*1024 , 4051e6 },
             { 64*1024*1024 , 4366e6 },
             { 128*1024*1024 , 5271e6 },
             { 256*1024*1024 , 5471e6 },
             { 512*1024*1024 , 6542e6 }
         };
      }
   }

   // page 4 in 20260916_IO_BW_measurements_SETONIX.odt
   if (strcasecmp(system_name,"setonix-ssd") == 0 ) {
      printf("BW INFO : using BW measured on SETONIX (SSD/flash partition)\n");

      if (use_random_read_write) {
         writeBW.bandwidthCurve = { // SSD/randwrite_bw_vs_size.txt
            { 4*1024 , 7.26e6 },
            { 64*1024 , 94.3e6 },
            { 1024*1024 , 567e6 },
            { 16*1024*1024 , 1616e6 },
            { 32*1024*1024 , 2047e6 },
            { 64*1024*1024 , 2250e6 },
            { 128*1024*1024 , 2472e6 },
            { 256*1024*1024 , 2568e6 },
            { 512*1024*1024 , 2545e6 },         
         };
      } else {
         writeBW.bandwidthCurve = { // SSD/write_bw_vs_size.txt
            { 4*1024 , 11.8e6 },
            { 64*1024 , 88.0e6 },
            { 1024*1024 , 558e6 },
            { 16*1024*1024 , 1590e6 },
            { 32*1024*1024 , 2003e6 },
            { 64*1024*1024 , 2204e6 },
            { 128*1024*1024 , 2429e6 },
            { 256*1024*1024 , 2415e6 },
            { 512*1024*1024 , 2512e6 }
         };
      }
      if (use_random_read_write) {
         readBW.bandwidthCurve = { // SSD/randread_bw_vs_size.cc
            { 4*1024 , 19.6e6 },
            { 64*1024 , 149e6 },
            { 1024*1024 , 755e6 },
            { 16*1024*1024 , 2091e6 },
            { 32*1024*1024 , 2123e6 },
            { 64*1024*1024 , 2391e6 },
            { 128*1024*1024 , 2374e6 },
            { 256*1024*1024 , 2587e6 },
            { 512*1024*1024 , 2785e6 }
         };
      } else {
         readBW.bandwidthCurve = { // SSD/read_bw_vs_size.cc
            { 4*1024 , 17.8e6 },
            { 64*1024 , 127e6 },
            { 1024*1024 , 464e6 },
            { 16*1024*1024 , 1546e6 },
            { 32*1024*1024 , 1784e6 },
            { 64*1024*1024 , 2072e6 },
            { 128*1024*1024 , 2294e6 },
            { 256*1024*1024 , 2482e6 },
            { 512*1024*1024 , 2569e6 }
         };
      }

   }

   // if specific system was not found above use the defaults below:  
   // LAPTOP BENCHMARKS on EXTERNAL HDD (see 20260916_IO_BW_measurements.odt) - these are used as default:
   // SEQUENTIAL WRITE (bytes written where the previous write finished)
   // IOCostModel writeBW;
   writeBW.bandwidthCurve = {
       {4*1024,       28.6e6},    // was 20e6,  fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_4k.out 2>&1
       {64*1024,      93.8e6},    // was 150e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64k --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_64k.out 2>&1
       {1024*1024,    93.0e6},    // was 600e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=1M --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_1M.out 2>&1
       {16*1024*1024, 94.3e6},    // was 1100e6,fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=16M --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_16M.out 2>&1 
       {32*1024*1024, 93.1e6},    // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=32M --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_32M.out 2>&1
       {64*1024*1024, 93.8e6},    // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64M --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_64M.out 2>&1
       {128*1024*1024,93.1e6}     // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=128M --rw=write --direct=1 --ioengine=libaio --runtime=300 --time_based > write_128M.out 2>&1 
   };
   
   if (use_random_read_write) {
      // RANDOM WRITES (bytes written at random position of the file)
      // IOCostModel writeBW;
      writeBW.bandwidthCurve = {
          {4*1024,      2.556e6}, // was 20e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_4k.out 2>&1
          {64*1024,      22.8e6}, // was 150e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64k --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_64k.out 2>&1
          {1024*1024,    80.4e6}, // was 600e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=1M --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_1M.out 2>&1
          {16*1024*1024, 70.0e6}, // was 1100e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=16M --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_16M.out 2>&1
          {32*1024*1024, 76.3e6}, // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=32M --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_32M.out 2>&1
          {64*1024*1024, 82.0e6}, // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64M --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_64M.out 2>&1
          {128*1024*1024,80.9e6}  // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=128M --rw=randwrite --direct=1 --ioengine=libaio --runtime=300 --time_based > randwrite_128M.out 2>&1
      };
   }

   // LAPTOP BENCHMARKS on EXTERNAL HDD:
   // SEQUENTIAL READ (bytes read where the previous write finished)
   // IOCostModel readBW;
   readBW.bandwidthCurve = {
       {4*1024,       26.1e6},    // was 20e6,  fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_4k.out 2>&1
       {64*1024,      91.9e6},    // was 150e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64k --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_64k.out 2>&1 
       {1024*1024,    92.4e6},    // was 600e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=1M --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_1M.out 2>&1
       {16*1024*1024, 88.8e6},    // was 1100e6,fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=16M --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_16M.out 2>&1
       {32*1024*1024, 90.2e6},    // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=32M --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_32M.out 2>&1
       {64*1024*1024, 88.5e6},    // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64M --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_64M.out 2>&1
       {128*1024*1024,91.4e6}     // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=128M --rw=read --direct=1 --ioengine=libaio --runtime=300 --time_based > read_128M.out 2>&1
   };
   
   if (use_random_read_write) {
      // RANDOM READS (bytes read at random position of the file)
      // IOCostModel readBW;
      readBW.bandwidthCurve = {
          {4*1024,      0.384e6}, // was 20e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=4k --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_4k.out 2>&1
          {64*1024,      6.541e6}, // was 150e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64k --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_64k.out 2>&1
          {1024*1024,    48.3e6}, // was 600e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=1M --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_1M.out 2>&1
          {16*1024*1024, 85.7e6}, // was 1100e6, fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=16M --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_16M.out 2>&1
          {32*1024*1024, 86.4e6}, // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=32M --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_32M.out 2>&1
          {64*1024*1024, 86.0e6}, // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=64M --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_64M.out 2>&
          {128*1024*1024,90.6e6}  //  // fio --name=iotest --filename=./fio_benchmark_scratch.dat --size=1G --bs=128M --rw=randread --direct=1 --ioengine=libaio --runtime=300 --time_based > randread_128M.out 2>&1
      };
   }
    
   return true;
}


void addTiledRotationPhases(IOCostBreakdown& result,
                             hsize_t stokes, hsize_t depth, hsize_t height, hsize_t width,
                             const IOCostModel& readModel, const IOCostModel& writeModel) {
    if (depth <= 1) return; // no rotation pass at all in this case

    std::vector<hsize_t> standardDims = {stokes, depth, height, width};
    std::vector<hsize_t> chunkDims    = {1, 1, TILE_SIZE, TILE_SIZE};
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
