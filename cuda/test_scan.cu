//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include <iostream>
#include "util/utility.cuh"
#include "cuda_base.cuh"
#include "params.h"
#include "CUDAStat.cuh"
#include <cub/cub.cuh>
using namespace std;

/* Test CUB scan */
bool test_scan(uint64_t len, CUDATimeStat *timing) {
    log_info("----------- Function: %s -----------", __FUNCTION__);
    log_info("Data cardinality=%d (%.1f MB)", len, 1.0*len* sizeof(int)/1024/1024);
    bool res = true;
    float ave_time = 0.0f;

    int *h_in_gpu = new int[len];
    int *h_in_cpu = new int[len];
#pragma omp parallel for
    for(int i = 0; i < len; i++) {
        h_in_gpu[i] = 1;
        h_in_cpu[i] = 1;
    }
    int *d_in;
    checkCudaErrors(cudaMalloc((void**)&d_in,sizeof(int)*len));
    cudaMemcpy(d_in, h_in_gpu, sizeof(int) * len, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        float cur_time;
        int *d_out;
        checkCudaErrors(cudaMalloc((void**)&d_out,sizeof(int)*len));

        // Allocate temporary storage
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        cudaEventRecord(start, 0);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len);
        checkCudaErrors(cudaMalloc(&d_temp_storage,temp_storage_bytes));
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&cur_time, start, end);

        if(e==0) {      //check
            cudaMemcpy(h_in_gpu, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);
            for(int i = 0; i < len; i++) {
                if (h_in_gpu[i] != i) {
                    res = false;
                }
            }
        }
        else if (res == true) {
            ave_time += cur_time;
        }
        else {
            log_error("Wrong results");
            res = false;
            break;
        }

        checkCudaErrors(cudaFree(d_out));
        checkCudaErrors(cudaFree(d_temp_storage));
    }
    ave_time/= (EXPERIMENT_TIMES-1);
    checkCudaErrors(cudaFree(d_in));

    delete[] h_in_gpu;
    delete[] h_in_cpu;

    log_info("Time=%.1f ms, throughput=%.1f GB/s", ave_time, compute_bandwidth(len, sizeof(int), ave_time));
    return res;
}

int main(int argc, char *argv[]) {
    cudaSetDevice(DEVICE_ID);
    CUDATimeStat timing;
    for(int scale = 10; scale <= 30; scale++) {
        uint64_t num = pow(2,scale);
        assert(test_scan(num, &timing));
    }
    return 0;
}