//
//  split.cu
//  comparison_gpu
//
//  Created by Bryan on 01/24/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

__global__
void createHist( const Record* source,
                        int length,
                        int *his,
                        int fanout)
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
        int offset = source[pos].y;
        temp[offset * localSize + localId]++;
    }
    
    for(int pos = 0; pos < fanout; pos ++) {
        his[pos * globalSize + globalId] = temp[pos * localSize + localId];
    }
}

__global__
void splitWithHist(const Record *source,
                   int* his,
                   int length,
                   Record *dest,
				   int fanout)
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
        int offset = source[pos].y;
        dest[temp[offset * localSize + localId]++] = source[pos];
    }
}

double splitDevice(Record *d_source, Record *d_dest, int* d_his, int r_len, int fanout, int blockSize, int gridSize) {
	double totalTime = 0.0f;

	dim3 grid(gridSize);
	dim3 block(blockSize);

	int globalSize = blockSize * gridSize;
	int hisLength = globalSize * fanout;

	struct timeval start, end;

	gettimeofday(&start, NULL);
	createHist<<<grid, block, sizeof(int)*fanout*blockSize>>>(d_source,r_len,d_his,fanout);
	scanDevice(d_his,hisLength, blockSize, gridSize, 1);
	splitWithHist<<<grid, block, sizeof(int)*fanout*blockSize>>>(d_source, d_his, r_len, d_dest, fanout);
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}

double splitImpl(Record *h_source, Record *h_dest, int r_len, int fanout, int blockSize, int gridSize) {
	
	double totalTime = 0.0f;

	int globalSize = blockSize * gridSize;
	int *d_his;
	Record *d_source, *d_dest;
	checkCudaErrors(cudaMalloc(&d_source, sizeof(Record)* r_len));
	checkCudaErrors(cudaMalloc(&d_dest, sizeof(Record)* r_len));
	checkCudaErrors(cudaMalloc(&d_his, sizeof(int)* globalSize * fanout));
	cudaMemcpy(d_source, h_source, sizeof(Record)*r_len, cudaMemcpyHostToDevice);

	totalTime = splitDevice(d_source, d_dest, d_his, r_len, fanout, blockSize, gridSize);
	
	cudaMemcpy(h_dest, d_dest, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	

	return totalTime;

}



