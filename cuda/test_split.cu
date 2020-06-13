//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include <cub/cub.cuh>
#include <iostream>
#include <algorithm>
#include "util/utility.cuh"
#include "cuda_base.cuh"
#include "params.h"
#include "CUDAStat.cuh"

using namespace std;

/*
 * GPU multisplit
 * Threads in a warp compute the histogram, each thread only knows the bucket_id of its element*/
__device__
void warp_histogram(int bucket_id, int bucket_bits, int *histo) {
    int localId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    int warpNum = globalSize / WARP_SIZE;
    int warpId = globalId / WARP_SIZE;
    int lane = localId & (WARP_SIZE-1);

    /*process buckets with multiple passes*/
    int buckets = 1<<bucket_bits;
    int rounds = (buckets + WARP_SIZE - 1) / WARP_SIZE; //deal with mutliple buckets
    unsigned histo_bmp[8];              //deal with at most 256 buckets.
    for(int k = 0; k < rounds; k++) histo_bmp[k] = 0xffffffff;
    for(int k = 0; k < bucket_bits; k++) {
        unsigned temp_buffer = __ballot_sync(0xffffffff, bucket_id & 0x01);
        for(int j = 0; j < rounds; j++) {
            if (((j*WARP_SIZE+lane)>>k) & 0x01)   histo_bmp[j] &= temp_buffer;
            else                                  histo_bmp[j] &= (0xffffffff ^ temp_buffer);
        }
        bucket_id >>= 1;
    }
    for(int j = 0; j < rounds; j++) {
        int idx = j * WARP_SIZE + lane;
        if (idx < buckets)  histo[idx * warpNum + warpId] = __popc(histo_bmp[j]);
    }

    /*process buckets with a single pass (<= 32 buckets) */
//    unsigned histo_bmp = 0xffffffff;              //deal with at most 256 buckets.
//    for(int k = 0; k < bucket_bits; k++) {
//        unsigned temp_buffer = __ballot_sync(0xffffffff, bucket_id & 0x01);
//        if ((lane>>k) & 0x01)   histo_bmp &= temp_buffer;
//        else                    histo_bmp &= (0xffffffff ^ temp_buffer);
//        bucket_id >>= 1;
//    }
//    histo[lane * warpNum + warpId] = __popc(histo_bmp);
}

__device__ void warp_offset(int *key_in, int *key_out, int bucket_id, int bucket_bits, int *histo) {
    int localId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    int warpNum = globalSize / WARP_SIZE;
    int warpId = globalId / WARP_SIZE;
    int lane = localId & (WARP_SIZE-1);
    int bucket_id_fixed = bucket_id;
    unsigned offset_bmp = 0xffffffff;              //deal with at most 256 buckets.

    for(int k = 0; k < bucket_bits; k++) {
        unsigned temp_buffer = __ballot_sync(0xffffffff, bucket_id & 0x01);
        if (bucket_id & 0x01)   offset_bmp &= temp_buffer;
        else                    offset_bmp &= (0xffffffff ^ temp_buffer);
        bucket_id >>= 1;
    }

    int offset = __popc(offset_bmp & (0xffffffff>>(31-lane)))-1;
    int pos = histo[bucket_id_fixed*warpNum + warpId]+offset;
    key_out[pos] = key_in[globalId];
}

__global__ void pre_scan(int *key_in, int *histo, int length, int bucket_bits) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int bucket_id = key_in[globalId];
    warp_histogram(bucket_id, bucket_bits, histo);
}

__global__ void post_scan(int *key_in, int *key_out, int *histo_scanned, int length, int bucket_bits) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int bucket_id = key_in[globalId];
    warp_offset(key_in, key_out, bucket_id, bucket_bits, histo_scanned);
}

//testing
//__global__ void transpose1(int *his_in, int *his_out,int buckets, int warp_num) {
//    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//    if (globalId == 0) {
//        for(int w = 0; w < warp_num; w++) {
//            for(int b = 0; b < buckets; b++) {
//                his_out[b*warp_num+w] = his_in[w*buckets+b];
//            }
//        }
//    }
//}
//
//__global__ void transpose2(int *his_in, int *his_out,int buckets, int warp_num) {
//    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//    if (globalId == 0) {
//        for(int b = 0; b < buckets; b++) {
//            for(int w = 0; w < warp_num; w++) {
//                his_out[w*buckets+b] = his_in[b*warp_num+w];
//            }
//        }
//    }
//}

bool test_split(uint64_t length, CUDATimeStat *timing) {
    log_info("----------- Function: %s -----------", __FUNCTION__);
    log_info("Data cardinality=%d (%.1f MB)", length, 1.0*length* sizeof(int)/1024/1024);
    auto block_size = 256;
    auto grid_size = (length + block_size - 1)/block_size;
    auto num_warps = block_size * grid_size / WARP_SIZE;
    float totalTime = 0;

    int bucket_bits = 5;        //32 buckets
    int buckets = 1<<bucket_bits;
    int *key_in = new int[length];
    int *key_out = new int[length];

    int *histograms = new int[buckets*num_warps];

    int *value_in = new int[length];
    int *value_out = new int[length];

    srand(time(nullptr));
    for(int i = 0; i <length; i++) {
        key_in[i] = rand() & (buckets-1);
    }

    int *d_key_in, *d_key_out, *d_histograms, *d_histograms_scanned;
    checkCudaErrors(cudaMalloc((void**)&d_key_in,sizeof(int)*length));
    checkCudaErrors(cudaMalloc((void**)&d_key_out,sizeof(int)*length));
    checkCudaErrors(cudaMalloc((void**)&d_histograms,sizeof(int)*buckets*num_warps));
    checkCudaErrors(cudaMalloc((void**)&d_histograms_scanned,sizeof(int)*buckets*num_warps));

    cudaMemcpy(d_key_in, key_in, sizeof(int) * length, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float tempTime;

    //1.pre-scan
    cudaEventRecord(start, 0);
    pre_scan<<<grid_size, block_size>>>(d_key_in, d_histograms, length, bucket_bits);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    log_info("Pre-scan time: %.1f ms", tempTime);
    totalTime += tempTime;

//    transpose1<<<1,1>>>(d_histograms, d_histograms_scanned, buckets, warp_num);

    //2.exclusive scan
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaEventRecord(start, 0);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histograms, d_histograms_scanned, buckets*num_warps);
    checkCudaErrors(cudaMalloc(&d_temp_storage,temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histograms, d_histograms_scanned, buckets*num_warps);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    log_info("Scan time: %.1f ms", tempTime);
    totalTime += tempTime;

    //test
//    transpose2<<<1,1>>>(d_histograms, d_histograms_scanned, buckets, warp_num);

    //3.post-scan
    cudaEventRecord(start, 0);
    post_scan<<<grid_size, block_size>>>(d_key_in, d_key_out, d_histograms_scanned, length, bucket_bits);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    log_info("Post-scan time: %.1f ms", tempTime);
    totalTime += tempTime;
    log_info("Total time: %.1f ms, throughput: %.1f GB/s",
             totalTime, compute_bandwidth(length, sizeof(int), totalTime));

    cudaMemcpy(histograms, d_histograms, sizeof(int) * WARP_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(key_out, d_key_out, sizeof(int) * length, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaFree(d_key_in));
    checkCudaErrors(cudaFree(d_key_out));
    checkCudaErrors(cudaFree(d_histograms));
    checkCudaErrors(cudaFree(d_histograms_scanned));

    //check
    sort(key_in, key_in+length);
    bool res = true;
    for(int i = 0; i <length; i++) {
        if (key_in[i] != key_out[i])    {
            res = false;
            break;
        }
    }
    if (!res) log_error("Wrong results");

    delete[] key_in;
    delete[] key_out;
    delete[] value_in;
    delete[] value_out;
    delete[] histograms;

    return res;
}

int main(int argc, char* argv[]) {
    cudaSetDevice(DEVICE_ID);
    CUDATimeStat timing;
    assert(test_split(stoll(argv[1]), &timing));
    return 0;
}