cmake_minimum_required(VERSION 3.0)
project(hdf_convert)

set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(HDF5)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(LINK_LIBS ${LINK_LIBS} cfitsio hdf5_serial hdf5_cpp hdf5_hl_cpp)

add_executable(hdf_convert main.cc)
target_link_libraries(hdf_convert ${LINK_LIBS})