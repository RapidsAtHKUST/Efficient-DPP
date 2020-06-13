cmake_minimum_required(VERSION 3.17)
project(EffPrim_CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(util)
include_directories(cub)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

find_package(CUDA)
if (CUDA_FOUND)
    message("--- CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc) #for helper_cuda.h

    if (USE_ARCH)
        message(STATUS "User has specified ARCH: ${USE_ARCH}")
        set(ARCH -arch=${USE_ARCH})
    else() # checking the default arch of the current platform
        set(OUTPUTFILE ./cuda_script)
        set(CUDAFILE ${CMAKE_SOURCE_DIR}/check_cuda.cu)
        execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
        execute_process(COMMAND ${OUTPUTFILE}
                RESULT_VARIABLE CUDA_RETURN_CODE
                OUTPUT_VARIABLE ARCH)

        if(${CUDA_RETURN_CODE} EQUAL 0)
            set(CUDA_SUCCESS "TRUE")
        else()
            set(CUDA_SUCCESS "FALSE")
        endif()

        if (${CUDA_SUCCESS})
            message(STATUS "CUDA Default Architecture: ${ARCH}")
            message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
            message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
            message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
            message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
        else()
            message(WARNING ${ARCH})
        endif()
    endif ()

    message(STATUS "Architecture specified: ${ARCH}")

    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()

    cuda_add_compile_options("-std=c++11 -O3 -g -lineinfo ${ARCH}")
    cuda_add_compile_options("--expt-extended-lambda -Wno-deprecated-gpu-targets")
    cuda_add_compile_options("-DUSE_LOG")
    cuda_add_compile_options("-Xcompiler=\"-fopenmp\"")
    cuda_add_compile_options("-Xcompiler=\"-Wno-deprecated\"")
    cuda_add_compile_options("-DDEVICE_ID=1")           #set device id

    cuda_add_executable(test_bandwidth test_bandwidth.cu) # test bandwidth
    cuda_add_executable(test_gather test_gather.cu) # test gather
    cuda_add_executable(test_scatter test_scatter.cu) # test scatter
    cuda_add_executable(test_scan test_scan.cu) # test scan
    cuda_add_executable(test_split test_split.cu) # test split
endif()