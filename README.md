# fits2idia

C++ implementation of FITS to IDIA-HDF5 converter, optimised using OpenMP

## Installation

Dependencies: CFITSIO, HDF5, HDF5 C++ bindings

To install:

    mkdir -p build
    cd build
    cmake ..
    make

## Commandline options

Run the executable with no parameters to see a list of options. The most 
important are:

```
-o      Output filename
-s      Use slower but less memory-intensive method (enable if memory allocation fails)
-p      Print progress output (by default the program is silent)
-m      Report predicted memory usage and exit without performing the conversion
```

## Configuration

A system administrator may set a memory usage limit in the `/etc/fits2idiarc`
configuration file. The executable will not attempt to convert a file if the
predicted memory usage exceeds this limit. A value of `0` means that there is no
limit. Use a very small value (like `1`) to disable all conversions.

An example configuration file is provided in the `static` directory, and is 
installed by the Ubuntu package to `usr/share/doc/fits2idia/examples`.
