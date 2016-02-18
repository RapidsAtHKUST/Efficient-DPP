//
//  scatter.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

__global__
void scatter(const Record *d_source,
			Record *d_res,
			const int r_len,
			const int *loc)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
		// int loc_temp = loc[threadId];
		d_res[loc[threadId]] = d_source[threadId];
		// d_res[loc[threadId]].y = d_source[threadId].y;
		threadId += threadNum;
	}
}

__global__
void scatter_int(int *d_source,
			int *d_res,
			const int r_len,
			const int *loc)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
		// int loc_temp = loc[threadId];
		d_res[loc[threadId]] = d_source[threadId];
		// d_res[loc[threadId]].y = d_source[threadId].y;
		threadId += threadNum;
	}
}

double scatterDevice(Record *d_source, Record *d_res, int r_len,int *d_loc, int blockSize, int gridSize) {
	dim3 grid(gridSize);
	dim3 block(blockSize);

	double totalTime = 0.0f;
	struct timeval start, end;

	gettimeofday(&start, NULL);	
	cudaDeviceSynchronize();
	scatter<<<grid, block>>>(d_source, d_res, r_len, d_loc);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}

double scatterDevice_int(int *d_source, int *d_res, int r_len,int *d_loc, int blockSize, int gridSize) {
	dim3 grid(gridSize);
	dim3 block(blockSize);

	double totalTime = 0.0f;
	struct timeval start, end;

	gettimeofday(&start, NULL);	
	cudaDeviceSynchronize();
	scatter_int<<<grid, block>>>(d_source, d_res, r_len, d_loc);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}

double scatterImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize) {
	
	Record *d_source, *d_res;
	int *d_loc;
	double totalTime = 0.0f;

	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_loc,sizeof(int)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_loc, h_loc, sizeof(int) * r_len, cudaMemcpyHostToDevice);

	totalTime = scatterDevice(d_source, d_res, r_len, d_loc, blockSize, gridSize);
	
	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));
	checkCudaErrors(cudaFree(d_loc));

	return totalTime;
}