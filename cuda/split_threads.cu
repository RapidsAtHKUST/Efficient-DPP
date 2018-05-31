/*
 * compile:
 * nvcc -o split_threads -O3 -arch=sm_35 split_threads.cu -I /usr/local/cuda/samples/common/inc/ -I.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>
#include <cub/cub/device/device_scan.cuh>
using namespace std;
using namespace cub;

#define WARP_SIZE   (32)

//threads in a warp compute the histogram, each thread only knows the bucket_id of its element
__device__ void warp_histogram(int *d_key_in, int bucket_bits, int *histo) {
//    int localId = threadIdx.x;
//    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//    int globalSize = blockDim.x * gridDim.x;
//    int warpNum = globalSize / WARP_SIZE;
//    int warpId = globalId / WARP_SIZE;
//    int lane = localId & (WARP_SIZE-1);

//    int buckets = 1<<bucket_bits;

//    int rounds = (buckets + WARP_SIZE - 1) / WARP_SIZE; //deal with mutliple buckets
//    unsigned histo_bmp[8];              //deal with at most 256 buckets.
//    for(int k = 0; k < rounds; k++) histo_bmp[k] = 0xffffffff;
//
//    for(int k = 0; k < bucket_bits; k++) {
//        unsigned temp_buffer = __ballot(bucket_id & 0x01);
//        for(int j = 0; j < rounds; j++) {
//            if (((j*WARP_SIZE+lane)>>k) & 0x01)   histo_bmp[j] &= temp_buffer;
//            else                                  histo_bmp[j] &= (0xffffffff ^ temp_buffer);
//        }
//        bucket_id >>= 1;
//    }
//
//    for(int j = 0; j < rounds; j++) {
//        int idx = j * WARP_SIZE + lane;
//        if (idx < buckets)  histo[idx * warpNum + warpId] = __popc(histo_bmp[j]);
//    }

    //simplified version (<= 32 buckets)
    unsigned histo_bmp = 0xffffffff;              //deal with at most 256 buckets.
    for(int k = 0; k < bucket_bits; k++) {
        unsigned temp_buffer = __ballot(bucket_id & 0x01);
        if ((lane>>k) & 0x01)   histo_bmp &= temp_buffer;
        else                    histo_bmp &= (0xffffffff ^ temp_buffer);
        bucket_id >>= 1;
    }
    histo[lane * warpNum + warpId] = __popc(histo_bmp);
////    histo[warpId*buckets + lane] = __popc(histo_bmp);
}

__device__ void warp_offset(int *key_in, int *key_out, int bucket_id, int bucket_bits, int *histo) {
    int localId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    int warpNum = globalSize / WARP_SIZE;
    int warpId = globalId / WARP_SIZE;
    int lane = localId & (WARP_SIZE-1);
    int bucket_id_fixed = bucket_id;
//    int buckets = 1<< bucket_bits;
    unsigned offset_bmp = 0xffffffff;              //deal with at most 256 buckets.

    for(int k = 0; k < bucket_bits; k++) {
        unsigned temp_buffer = __ballot(bucket_id & 0x01);
        if (bucket_id & 0x01)   offset_bmp &= temp_buffer;
        else                    offset_bmp &= (0xffffffff ^ temp_buffer);
        bucket_id >>= 1;
    }

    int offset = __popc(offset_bmp & (0xffffffff>>(31-lane)))-1;
    int pos = histo[bucket_id_fixed*warpNum + warpId]+offset;
//    int pos = histo[warpId * buckets + bucket_id_fixed] + offset;
    key_out[pos] = key_in[globalId];
}

__global__ void pre_scan(int *key_in, int *histo, int length, int bucket_bits) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//    int bucket_id = key_in[globalId];
    warp_histogram(key_in, bucket_bits, histo);
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

int main() {

    int length = 1<<25;
    int local_size = 256;
    int grid_size = length/local_size;
    int warp_num = local_size * grid_size / WARP_SIZE;
    float totalTime = 0;

    int bucket_bits = 5;        //32 buckets
    int buckets = 1<<bucket_bits;
    int *key_in = new int[length];
    int *key_out = new int[length];

    int *histograms = new int[buckets*warp_num];

    int *value_in = new int[length];
    int *value_out = new int[length];

    srand(time(NULL));
    for(int i = 0; i <length; i++) {
        key_in[i] = rand() & (buckets-1);
    }

    int *d_key_in, *d_key_out, *d_histograms, *d_histograms_scanned;
    checkCudaErrors(cudaMalloc(&d_key_in,sizeof(int)*length));
    checkCudaErrors(cudaMalloc(&d_key_out,sizeof(int)*length));
    checkCudaErrors(cudaMalloc(&d_histograms,sizeof(int)*buckets*warp_num));
    checkCudaErrors(cudaMalloc(&d_histograms_scanned,sizeof(int)*buckets*warp_num));

    cudaMemcpy(d_key_in, key_in, sizeof(int) * length, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float tempTime;

    //1.pre-scan
    cudaEventRecord(start, 0);
    pre_scan<<<grid_size, local_size>>>(d_key_in, d_histograms, length, bucket_bits);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    cout<<"Pre-scan time: "<<tempTime<<" ms."<<endl;
    totalTime += tempTime;

//    transpose1<<<1,1>>>(d_histograms, d_histograms_scanned, buckets, warp_num);

    //2.exclusive scan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaEventRecord(start, 0);

    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histograms, d_histograms_scanned, buckets*warp_num));
    checkCudaErrors(cudaMalloc(&d_temp_storage,temp_storage_bytes));

    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histograms, d_histograms_scanned, buckets*warp_num));
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    cout<<"Scan time: "<<tempTime<<" ms."<<endl;
    totalTime += tempTime;


    //test
//    transpose2<<<1,1>>>(d_histograms, d_histograms_scanned, buckets, warp_num);

    //3.post-scan
    cudaEventRecord(start, 0);
    post_scan<<<grid_size, local_size>>>(d_key_in, d_key_out, d_histograms_scanned, length, bucket_bits);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&tempTime, start, end);
    cout<<"Post-scan time: "<<tempTime<<" ms."<<endl;
    totalTime += tempTime;

    cout<<"Total time: "<<totalTime<<" ms."<<endl;

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
            cout<<key_in[i]<<' '<<key_out[i]<<endl;
            break;
        }
    }
    if (res)    cout<<"Res: correct!"<<endl;
    else        cout<<"Res: wrong!"<<endl;

    delete[] key_in;
    delete[] key_out;
    delete[] value_in;
    delete[] value_out;
    delete[] histograms;

    return 0;
}