cmake_minimum_required(VERSION 3.1)
project(HeteroSys_study)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g")

set(UTIL_DIR ${CMAKE_SOURCE_DIR}/util)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(IMPL_DIR ${CMAKE_SOURCE_DIR}/Impl)
set(HEADER_DIR ${CMAKE_SOURCE_DIR}/inc)

# Moving executable files into bin
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

#include paths
include_directories(inc)

# Add all the header files automatically
file(GLOB_RECURSE HEADERS ${HEADER_DIR}/*)

# Add all the source files automatically
file(GLOB_RECURSE SOURCE_FILES ${UTIL_DIR}/* ${DATA_GENERATOR_DIR}/*.cu ${CMAKE_SOURCE_DIR}/third_party/topkapi/*.cpp)

file(GLOB_RECURSE SRC_FILES ${UTIL_DIR}/* ${IMPL_DIR}/*)

#todo: currently only the main, to be revised later
# Add all the test files automatically
file(GLOB_RECURSE TEST_FILES ${TEST_DIR}/*)

find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})

    # Generate the target name and add to executable
    foreach(test_file ${TEST_FILES})
        file(RELATIVE_PATH relative_name ${TEST_DIR} ${test_file}) #get relative path
        string(REPLACE ".cpp" "" target_name ${relative_name} ) #remove the suffix

        #compile options
        add_compile_options("-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"")

        add_executable(${target_name} ${test_file} ${SRC_FILES})
        target_link_libraries(${target_name} ${OpenCL_LIBRARY})
    endforeach()
endif()

#find_package(OpenCL)
#if (OpenCL_FOUND)
#    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
#
#    set(ARCH "sm_61" CACHE STRING "setting the architecture code") #default arch code
#    if(${CUDA_PREFETCH}) #specify
#        message("CUDA prefetching enabled.")
#        option(PREFETCH "CUDA prefetching" ON)
#    endif()
#
#    #for CUDA new functions later than 9.0
#    if ((CUDA_VERSION VERSION_EQUAL "9.0") OR (CUDA_VERSION VERSION_GREATER "9.0"))
#        option(CUDA_9_LATER "using CUDA 9 or later" ON)
#    endif()
#
#    macro(cuda_add_compile_options options)
#        list(APPEND CUDA_NVCC_FLAGS " ${options}")
#    endmacro()
#
#    #cmake config file for project
#    configure_file("${CMAKE_SOURCE_DIR}/config.h.in" "${CMAKE_BINARY_DIR}/config.h")
#    include_directories("${CMAKE_BINARY_DIR}")
#    include_directories("${CMAKE_SOURCE_DIR}")
#
#    #define the ARCH_35 or ARCH_61 macro
#    message("Architecture used: ${ARCH}")
#    if(${ARCH} STREQUAL "sm_35")
#        cuda_add_compile_options("-DARCH_35")
#    elseif(${ARCH} STREQUAL "sm_61")
#        cuda_add_compile_options("-DARCH_61")
#    endif()
#
#    cuda_add_compile_options("-g")
#    cuda_add_compile_options("-arch=${ARCH}")
#    cuda_add_compile_options("-Xcompiler=\"-Wno-deprecated\"")
#    #    cuda_add_compile_options("-Xptxas -v")     #list the usage of registers and memory
#    cuda_add_compile_options("-std=c++11 -O3 -g -lineinfo")
#    cuda_add_compile_options("-Xcompiler=\"-fopenmp\"")
#    cuda_add_compile_options("-DDBADDR=\"${DATA_DIR}\"")
#    cuda_add_compile_options("-DUSE_LOG")
#    cuda_add_compile_options("--expt-extended-lambda -Wno-deprecated-gpu-targets")
#
#    # Build the project into a shared library and link it later
#    cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADERS})
#
#    # Generate the target name and add to executable
#    foreach(test_file ${TEST_FILES})
#        file(RELATIVE_PATH relative_name ${TEST_DIR} ${test_file}) #get relative path
#        string(REPLACE ".cu" "" target_name ${relative_name} ) #remove the suffix
#
#        cuda_add_executable(${target_name} ${test_file})
#        target_link_libraries(${target_name} ${PROJECT_NAME})
#    endforeach()
#
#endif()


