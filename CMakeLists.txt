cmake_minimum_required(VERSION 2.8)
project(fits2idia)

set(default_build_type "Release")

set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(HDF5 REQUIRED COMPONENTS CXX)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (Verbose)
    ADD_DEFINITIONS(-D_VERBOSE_)
endif()

if (Timer)
    ADD_DEFINITIONS(-D_TIMER_)
endif()

set(LINK_LIBS ${LINK_LIBS} cfitsio ${HDF5_LIBRARIES})

set(SOURCE_FILES
    ${SOURCE_FILES}
    main.cc
    Stats.cc
    MipMap.cc
    Converter.cc
    FastConverter.cc
    SlowConverter.cc
    Util.cc)

add_executable(fits2idia ${SOURCE_FILES})
target_link_libraries(fits2idia ${LINK_LIBS})
