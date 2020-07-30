#include "UI.h"
#include "Converter.h"

int main(int argc, char** argv) {
    std::string inputFileName;
    std::string outputFileName;
    bool slow(false);
    bool progress(false);
    bool onlyReportMemory(false);

    if (!getOptions(argc, argv, inputFileName, outputFileName, slow, progress, onlyReportMemory)) {
        return 1;
    }

    std::unique_ptr<Converter> converter;

    try {
        converter = Converter::getConverter(inputFileName, outputFileName, slow, progress);

        if (onlyReportMemory) {
            converter->reportMemoryUsage();
            return 0;
        }

        DEBUG(std::cout << "Converting FITS file " << inputFileName << " to HDF5 file " << outputFileName << (slow ? " using slower, memory-efficient method" : "") << std::endl;);

        converter->convert();
    } catch (const char* msg) {
        std::cerr << "Error: " << msg << ". Aborting." << std::endl;
        return 1;
    }

    return 0;
}
