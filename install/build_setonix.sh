#!/bin/bash -e

# WARNING : this script is assumed to be for a supercomputer (HPC) : Setonix

build_dir=build
cmake_options=""

build_opt="Release" # or  "Debug"
if [[ -n "$1" && "$1" != "-" ]]; then
   build_opt="$1"
fi

# default options decided based on build type can be over-written by the 2nd parameter:
if [[ -n "$2" && "$2" != "-" ]]; then
   cmake_options=$2
fi

version=""
if [[ -n "$3" && "$3" != "-" ]]; then
   version="$3"
   build_dir=${build_dir}_${version}
fi

invoke_dir=`dirname $0`
# First, you need to source the bash library
# module load bash-utils
echo "source ${invoke_dir}/build_utils.sh"
source "${invoke_dir}/build_utils.sh"


PROGRAM_NAME=fits2idia${version}
if [[ -n "$4" && "$4" != "-" ]]; then
   PROGRAM_NAME="$4"
fi

PROGRAM_VERSION=devel
if [[ -n "$5" && "$5" != "-" ]]; then
   PROGRAM_VERSION="$5"
fi

export PAWSEY_PROJECT=ja3
if [[ -n "$6" && "$6" != "-" ]]; then
   export PAWSEY_PROJECT="$6"
fi


echo "############################################"
echo "PARAMETERS (build.sh scripts) :"
echo "############################################"
echo "PROGRAM_NAME = $PROGRAM_NAME"
echo "PROGRAM_VERSION = $PROGRAM_VERSION"
echo "############################################"


 # the following function sets up the installation path according to the
# cluster the script is running on and the first argument given. The argument
# can be:
# - "group": install the software in the group wide directory
# - "user": install the software only for the current user
# - "test": install the software in the current working directory 
echo "process_build_script_input user"
process_build_script_input user 


# load all the modules required for the program to compile and run.
# the following command also adds those module names in the modulefile
# that this script will generate.
echo "Loading required modules ..."
echo "Loading modules for PAWSEY_CLUSTER = $PAWSEY_CLUSTER"
if [ $PAWSEY_CLUSTER = "setonix" ]; then
   module reset

   module_load cfitsio/4.4.0 cray-hdf5/1.14.3.7
   
   # cmake is only required at build time, so we use the normal module load
   module load cmake/3.30.5
else 
   echo "ERROR Currently only Setonix is handled correctly"
   exit -1
fi   
# build your software..
echo "Building the software.."

[ -d ${build_dir} ] || mkdir ${build_dir}
cd ${build_dir}
pwd
echo "cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_BUILD_TYPE=${build_opt} ${cmake_options}"
cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_BUILD_TYPE=${build_opt} ${cmake_options}
make VERBOSE=1

# Install the software
make install

# This may not be the most elegant way to do this, but will fix later:
echo "cp ../scripts/pawsey_submit_smart_converter.sh ${INSTALL_DIR}/bin/"
cp ../scripts/pawsey_submit_smart_converter.sh ${INSTALL_DIR}/bin/

# test:
# if [[ $dotests -gt 0 ]]; then
#   echo "make test"
#   make test
#else
#   echo "WARNING : tests are not required"
#fi   

echo "Create the modulefile in $MODULEFILE_DIR (or $INSTALL_DIR)"
# export ADDITIONAL_MODULEFILE_COMMANDS="prepend_path('BLINK_IMAGER_PATH', root_dir )"
create_modulefile

echo "Done."


