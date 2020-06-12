cmake_minimum_required(VERSION 3.1)
project(HeteroSys_study)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -w -fopenmp")

set(UTIL_DIR ${CMAKE_SOURCE_DIR}/util)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(IMPL_DIR ${CMAKE_SOURCE_DIR}/primitives)

#include paths
include_directories(util)

# Add all the source files automatically
file(GLOB_RECURSE SOURCE_FILES ${UTIL_DIR}/*)

file(GLOB_RECURSE SRC_FILES ${UTIL_DIR}/* ${IMPL_DIR}/*)

# Add all the test files automatically
file(GLOB_RECURSE TEST_FILES ${TEST_DIR}/*)

#We provide the OpenCL 1.2 header files
set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")

find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})

    # Generate the target name and add to executable
    foreach(test_file ${TEST_FILES})
        file(RELATIVE_PATH relative_name ${TEST_DIR} ${test_file}) #get relative path
        string(REPLACE ".cpp" "" target_name ${relative_name} ) #remove the suffix

        #compile options
        add_compile_options("-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"")
        add_compile_options("-DUSE_LOG")

        add_executable(${target_name} ${test_file} ${SRC_FILES})
        target_link_libraries(${target_name} ${OpenCL_LIBRARY})
    endforeach()
endif()


