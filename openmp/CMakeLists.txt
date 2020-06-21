cmake_minimum_required(VERSION 3.1)
project(EffPrim_OpenMP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -w -mavx2 -fopenmp -ltbb")

set(UTIL_DIR ${CMAKE_SOURCE_DIR}/util)

#include paths
include_directories(util)

# Add all the source files automatically
file(GLOB_RECURSE SRC_FILES ${UTIL_DIR}/*)

add_compile_options("-DUSE_LOG")
add_executable(test_bandwidth_CPU test_bandwidth_CPU.cpp ${SRC_FILES})
add_executable(test_gather_scatter_CPU test_gather_scatter_CPU.cpp ${SRC_FILES})
add_executable(test_scan_CPU test_scan_CPU.cpp ${SRC_FILES})



