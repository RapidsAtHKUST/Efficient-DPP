cmake_minimum_required(VERSION 3.1)
project(EffPrim_OpenMP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -w -mavx2 -fopenmp")

set(UTIL_DIR ${CMAKE_SOURCE_DIR}/util)

#include paths
include_directories(util)

# Add all the source files automatically
file(GLOB_RECURSE SRC_FILES ${UTIL_DIR}/*)

add_compile_options("-DUSE_LOG")
add_executable(test_bandwidth_CPU test_bandwidth_CPU.cpp ${SRC_FILES})
add_executable(test_gather_scatter_CPU test_gather_scatter_CPU.cpp ${SRC_FILES})

#    # Generate the target name and add to executable
#    foreach(test_file ${TEST_FILES})
#        file(RELATIVE_PATH relative_name ${TEST_DIR} ${test_file}) #get relative path
#        string(REPLACE ".cpp" "" target_name ${relative_name} ) #remove the suffix
#
#        #compile options
#        add_compile_options("-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"")
#        add_compile_options("-DUSE_LOG")
#
#        add_executable(${target_name} ${test_file} ${SRC_FILES})
#        target_link_libraries(${target_name} ${OpenCL_LIBRARY})
#    endforeach()



