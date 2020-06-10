//
//  split.cu
//  comparison_gpu
//
//  Created by Zhuohang Lai on 01/24/16.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#include "kernels.h"

template<class T>
__global__ void createHist( 
    T* d_source_values,
    int length, int *his, int fanout)
{
	extern __shared__ int temp[];		//size: fanout * localSize

    int localId = threadIdx.x;
    int localSize = blockDim.x;
    int globalId = blockDim.x * blockIdx.x + threadIdx.x;
    int globalSize = blockDim.x * gridDim.x;

    for(int pos = 0; pos < fanout; pos ++) {
        temp[pos * localSize + localId] = 0;
    }
    __syncthreads();

    for(int pos = globalId; pos < length; pos += globalSize) {
        int offset = (int)d_source_values[pos];
        assert(offset < fanout);
        temp[offset * localSize + localId]++;
    }
    
    for(int pos = 0; pos < fanout; pos ++) {
        his[pos * globalSize + globalId] = temp[pos * localSize + localId];
    }
}

template<class T>
__global__ void splitWithHist(
#ifdef RECORDS
    int *d_source_keys, int *d_dest_keys, 
#endif
    T *d_source_values, T *d_dest_values,
    int* his, int length, int fanout
#ifdef RECORDS
    ,bool isRecord
#endif
    )
{
	extern __shared__ int temp[];			//size: fanout * localSize

    int localId = threadIdx.x;
    int localSize = blockDim.x;
    int globalId = blockDim.x * blockIdx.x + threadIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    for(int pos = 0; pos < fanout; pos ++) {
        temp[pos * localSize + localId] = his[pos* globalSize + globalId];
    }
    
    for(int pos = globalId; pos < length; pos += globalSize) {
        T offset = d_source_values[pos];
#ifdef RECORDS
        if (isRecord) 
            d_dest_keys[temp[offset * localSize + localId]] = d_source_keys[pos];
#endif        
        d_dest_values[temp[offset * localSize + localId]++] = d_source_values[pos];
    }
}

template<class T>
float split(
#ifdef RECORDS
    int *d_source_keys, int *d_dest_keys,
#endif
    T *d_source_values, T *d_dest_values, 
    int* d_his, int r_len, int fanout, int blockSize, int gridSize
#ifdef RECORDS
    ,bool isRecord
#endif
    ) 
{
	float totalTime = 0.0f;
    blockSize = 256;
	dim3 grid(gridSize);
	dim3 block(blockSize);

	int globalSize = blockSize * gridSize;
	int hisLength = globalSize * fanout;

	cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    SHARED_MEM_CHECK(sizeof(int) * fanout * blockSize)

    cudaEventRecord(start);
	createHist<T><<<grid, block, sizeof(int)*fanout*blockSize>>>(d_source_values, r_len,d_his,fanout);
	scan_ble<int>(d_his, hisLength, 1, 1024);
	splitWithHist<T><<<grid, block, sizeof(int)*fanout*blockSize>>>(
#ifdef RECORDS
    d_source_keys, d_dest_keys, 
#endif
    d_source_values,d_dest_values,
    d_his, r_len, fanout
#ifdef RECORDS
    ,isRecord
#endif
    );
	cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

//template
template float split<int>(
#ifdef RECORDS
    int *d_source_keys, int *d_dest_keys,
#endif
    int *d_source_values, int *d_dest_values, 
    int* d_his, int r_len, int fanout, int blockSize, int gridSize
#ifdef RECORDS
    ,bool isRecord
#endif
    ); 



