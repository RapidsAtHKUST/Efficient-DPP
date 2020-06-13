//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include <iostream>
#include "util/utility.cuh"
#include "cuda_base.cuh"
#include "params.h"
#include "CUDAStat.cuh"
using namespace std;

#define SCALAR  (3)

__global__
void copy_kernel(int *d_in, int *d_out, int num) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < num) {
        d_out[globalId] = d_in[globalId];
    }
}

__global__
void scale_kernel(int *d_in, int *d_out, int num) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < num) {
        d_out[globalId] = d_in[globalId] * SCALAR;
    }
}

__global__
void addition_kernel(int *d_in_1, int *d_in_2, int *d_out, int num) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < num) {
        d_out[globalId] = d_in_1[globalId] + d_in_2[globalId];
    }
}

__global__
void triad_kernel(int *d_in_1, int *d_in_2, int *d_out, int num) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < num) {
        d_out[globalId] = d_in_1[globalId] + SCALAR * d_in_2[globalId];
    }
}

void test_bandwidth(uint64_t len, CUDATimeStat *timing) {
    log_info("----------- Function: %s -----------", __FUNCTION__);
    log_info("Data cardinality=%d (%.1f MB)", len, 1.0*len* sizeof(int)/1024/1024);
    auto block_size = 256;
    auto grid_size = (len + block_size - 1) / block_size;
    float ave_time = 0.0;

    int *h_in_1, *d_in_1;
    int *h_in_2, *d_in_2;
    int *d_out;

    h_in_1 = new int[len];
    h_in_2 = new int[len];
    for(int i = 0; i < len; i++) {
        h_in_1[i] = i;
        h_in_2[i] = i + 10;
    }
    checkCudaErrors(cudaMalloc((void**)&d_in_1,sizeof(int)*len));
    checkCudaErrors(cudaMalloc((void**)&d_in_2,sizeof(int)*len));
    checkCudaErrors(cudaMalloc((void**)&d_out,sizeof(int)*len));
    cudaMemcpy(d_in_1, h_in_1, sizeof(int)*len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_2, h_in_2, sizeof(int)*len, cudaMemcpyHostToDevice);

    /*copy kernel*/
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        auto gpu_time_idx = timing->get_idx();
        execKernel(copy_kernel, grid_size, block_size, timing, false, d_in_1, d_out, len);
        if (i != 0)     ave_time += timing->diff_time(gpu_time_idx);
    }
    ave_time /= (EXPERIMENT_TIMES - 1);
    log_info("Copy: time=%.1f ms, throughput=%.1f GB/s",
    ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));

    /*scale kernel*/
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        auto gpu_time_idx = timing->get_idx();
        execKernel(scale_kernel, grid_size, block_size, timing, false, d_in_1, d_out, len);
        if (i != 0)     ave_time += timing->diff_time(gpu_time_idx);
    }
    ave_time /= (EXPERIMENT_TIMES - 1);
    log_info("Scale: time=%.1f ms, throughput=%.1f GB/s",
             ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));

    /*addition kernel*/
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        auto gpu_time_idx = timing->get_idx();
        execKernel(addition_kernel, grid_size, block_size, timing, false, d_in_1, d_in_2, d_out, len);
        if (i != 0)     ave_time += timing->diff_time(gpu_time_idx);
    }
    ave_time /= (EXPERIMENT_TIMES - 1);
    log_info("Addition: time=%.1f ms, throughput=%.1f GB/s",
             ave_time, compute_bandwidth(len*3, sizeof(int), ave_time));

    /*triad kernel*/
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        auto gpu_time_idx = timing->get_idx();
        execKernel(triad_kernel, grid_size, block_size, timing, false, d_in_1, d_in_2, d_out, len);
        if (i != 0)     ave_time += timing->diff_time(gpu_time_idx);
    }
    ave_time /= (EXPERIMENT_TIMES - 1);
    log_info("Triad: time=%.1f ms, throughput=%.1f GB/s",
             ave_time, compute_bandwidth(len*3, sizeof(int), ave_time));

    delete[] h_in_1;
    delete[] h_in_2;
    checkCudaErrors(cudaFree(d_in_1));
    checkCudaErrors(cudaFree(d_in_2));
    checkCudaErrors(cudaFree(d_out));
}

int main(int argc, char *argv[]) {
    CUDATimeStat timing;
    for(int scale = 10; scale <= 30; scale++) {
        uint64_t data_size = 1<<scale;
        test_bandwidth(data_size, &timing);
    }
    return 0;
}