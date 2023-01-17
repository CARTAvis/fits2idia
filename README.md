# fits2idia

C++ implementation of FITS to IDIA-HDF5 converter, optimised using OpenMP

## Installation

### Ubuntu package

Packages for the last two Ubuntu LTS releases are available from the [CARTA PPA](https://launchpad.net/~cartavis-team/+archive/ubuntu/carta). Currently Bionic (18.04) and Focal (20.04) are officially supported.

    sudo add-apt-repository ppa:cartavis-team/carta
    sudo apt update
    sudo apt install fits2idia

### RPM package

RPM packages are available from the [CARTA RPM package repository](https://packages.cartavis.org/). We officially support the latest versions of RHEL 7 or CentOS 7, and RHEL 8 or AlmaLinux 8 (or equivalent).

RHEL 7 / CentOS 7:

    sudo yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
    sudo yum-config-manager --add-repo https://packages.cartavis.org/cartavis.repo
    sudo yum -y install fits2idia

RHEL 8 / AlmaLinux 8:

    sudo yum -y install yum-utils
    sudo yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
    sudo yum-config-manager --enable powertools
    sudo yum-config-manager --add-repo https://packages.cartavis.org/cartavis.repo
    sudo yum -y install fits2idia

### macOS (using Homebrew)

    brew install cartavis/tap/fits2idia

### AppImage

AppImages built for the `x86_64` and `aarch64` architectures can be downloaded from the [latest release page](https://github.com/CARTAvis/fits2idia/releases/latest).

    wget https://github.com/CARTAvis/fits2idia/releases/latest/download/fits2idia.AppImage.`arch`.tgz
    tar -xzf fits2idia.AppImage.`arch`.tgz

### Building from source

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

## For developers

Automatically generated Doxygen documentation can be found at [cartavis.org/fits2idia](https://cartavis.org/fits2idia/).
