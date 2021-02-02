cmake_minimum_required(VERSION 3.10)
project(fits2idia)

set(default_build_type "Release")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

find_package(HDF5 REQUIRED COMPONENTS CXX)
if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
    set(LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES})
else ()
    message(FATAL_ERROR "Could not find CFITSIO.")
endif ()

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_SEARCH_MODULE(CFITSIO REQUIRED cfitsio)
if (CFITSIO_FOUND)
    include_directories(${CFITSIO_INCLUDE_DIRS})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    set(LINK_LIBS ${LINK_LIBS} ${CFITSIO_LIBRARIES})
    message(STATUS "Found CFITSIO: ${CFITSIO_LINK_LIBRARIES} Version: ${CFITSIO_VERSION}")
else ()
    message(FATAL_ERROR "Could not find CFITSIO.")
endif ()

if (Verbose)
    ADD_DEFINITIONS(-D_VERBOSE_)
endif ()

if (Timer)
    ADD_DEFINITIONS(-D_TIMER_)
endif ()

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

install(TARGETS fits2idia
    RUNTIME DESTINATION bin
)
