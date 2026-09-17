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

bool get_io_cost_model(const char* system_name, IOCostModel& setonixWrite, IOCostModel& setonixRead )
{
   // TODO: implement System-specific BWs, and use system_name. 
   //       for now all the same based measurements on my laptop

   // IOCostModel setonixWrite;
   setonixWrite.bandwidthCurve = {
       {4*1024,     20e6},   // fio --bs=4k  --rw=randwrite
       {64*1024,   150e6},   // fio --bs=64k --rw=randwrite
       {1024*1024, 600e6},   // fio --bs=1m  --rw=write
       {16*1024*1024, 1100e6} // fio --bs=16m --rw=write
   };    
       
   // IOCostModel setonixWrite;
   setonixRead.bandwidthCurve = {
       {4*1024,     20e6},   // fio --bs=4k  --rw=randwrite
       {64*1024,   150e6},   // fio --bs=64k --rw=randwrite
       {1024*1024, 600e6},   // fio --bs=1m  --rw=write
       {16*1024*1024, 1100e6} // fio --bs=16m --rw=write
   };

    
   // TODO: if (strcasecmp(system_name,"SETONIX") == 0 ){ return true; }  
    
   return true;
}