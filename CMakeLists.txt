cmake_minimum_required(VERSION 3.13)
project(hdf_convert)

set(default_build_type "Release")

set(CMAKE_CXX_STANDARD 11)

# macro(find_hdf5)
# 	find_package(HDF5 REQUIRED COMPONENTS C)
# 	if (HDF5_FOUND)
# 		include_directories(${HDF5_INCLUDE_DIRS})
# 		if (HDF5_IS_PARALLEL HDF5_CONVERTER_ALLOWPARALLELHDF5)
# 			set (ENV{HDF5_PREFER_PARALLEL} true)
# 			set(HDF5_CONVERTER_HAS_PARALLEL_HDF5 Yes)
# 			list(APPEND HDF5_CONVERTER_DEFINES USEPARALLELHDF)
# 			if (HDF5_VERSION VERSION_GREATER "1.10.0" AND HDF5_CONVERTER_ALLOWCOMPRESSIONPARALLELHDF5)
# 				set(HDF5_CONVERTER_HAS_COMPRESSED_HDF5 Yes)
# 				list(APPEND HDF5_CONVERTER_DEFINES USEHDFCOMPRESSION)
# 				list(APPEND HDF5_CONVERTER_DEFINES PARALLELCOMPRESSIONACTIVE)
# 			endif()
# 		else()
# 			if (HDF5_CONVERTER_ALLOWCOMPRESSIONHDF5)
# 				set(HDF5_CONVERTER_HAS_COMPRESSED_HDF5 Yes)
# 				list(APPEND HDF5_CONVERTER_DEFINES USEHDFCOMPRESSION)
# 			endif()
# 		endif()
#     endif()
# endmacro()

FIND_PACKAGE(HDF5 REQUIRED COMPONENTS C)
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
    HDF5Wrapper.cc
    Converter.cc
    FastConverter.cc
    SlowConverter.cc
    Stats.cc
    MipMap.cc
    Util.cc
    main.cc
    )

add_executable(hdf_convert ${SOURCE_FILES})
target_link_libraries(hdf_convert ${LINK_LIBS})
