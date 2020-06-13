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

__global__
void gather_kernel(int *d_in, int *d_out, int *d_idx, int num) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < num) {
        d_out[globalId] = d_in[d_idx[globalId]];
    }
}

void test_gather(int len, CUDATimeStat *timing) {
    log_info("----------- Function: %s -----------", __FUNCTION__);
    log_info("Data cardinality=%d (%.1f MB)", len, 1.0*len* sizeof(int)/1024/1024);
    auto block_size = 256;
    auto grid_size = (len + block_size - 1) / block_size;
    float ave_time = 0.0;

    int *h_in, *d_in, *d_out, *h_idx, *d_idx;
    h_in = new int[len];
    h_idx = new int[len];
#pragma omp parallel for
    for(int i = 0; i < len; i++) {
        h_in[i] = i;
    }

    srand((unsigned)time(nullptr));
    random_generator_int_unique(h_idx, len);
    checkCudaErrors(cudaMalloc((void**)&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc((void**)&d_out,sizeof(int)*len));
    checkCudaErrors(cudaMalloc((void**)&d_idx,sizeof(int)*len));
    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, sizeof(int)*len, cudaMemcpyHostToDevice);

    for(auto p = 1; p <= 32; p<<= 1) {
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            auto gpu_time_idx = timing->get_idx();
            execKernel(gather_kernel, grid_size, block_size, timing, false, d_in, d_out, d_idx, len);
            if (i != 0) ave_time += timing->diff_time(gpu_time_idx);
        }
        ave_time /= (EXPERIMENT_TIMES - 1);
        log_info("Pass=%d, time=%.1f ms, throughput=%.1f GB/s",
                 p, ave_time, compute_bandwidth(len, sizeof(int), ave_time));
    }

    delete[] h_in;
    delete[] h_idx;
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_idx));
    checkCudaErrors(cudaFree(d_out));
}

/*
 * ./test_gather DATA_SIZE
 * */
int main(int argc, char *argv[]) {
    assert(argc == 2);
    cudaSetDevice(DEVICE_ID);
    CUDATimeStat timing;
    test_gather(stoi(argv[1]), &timing);
    return 0;
}