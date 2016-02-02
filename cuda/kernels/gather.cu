//
//  gather.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

__global__
void gather(const Record *d_source,
			Record *d_res,
			const int r_len,
			const int *loc)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
		d_res[threadId] = d_source[loc[threadId]];
		threadId += threadNum;
	}
}

__global__
void gather_mul(const Record *d_source,
			Record *d_res,
			const int r_len,
			const int *loc,
			int from, 
			int to)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
		int tempLoc = loc[threadId];
		if (tempLoc >= from && tempLoc < to) {
			d_res[threadId] = d_source[tempLoc];
		}
		threadId += threadNum;
	}
}

double gatherDevice(Record *d_source, Record *d_res, int r_len,int *d_loc, int blockSize, int gridSize) {
	dim3 grid(gridSize);
	dim3 block(blockSize);

	double totalTime = 0.0f;
	struct timeval start, end;

	gettimeofday(&start, NULL);
	gather<<<grid, block>>>(d_source, d_res, r_len, d_loc);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}

double gatherDevice_mul(Record *d_source, Record *d_res, int r_len,int *d_loc, int blockSize, int gridSize, int from, int to) {
	dim3 grid(gridSize);
	dim3 block(blockSize);

	double totalTime = 0.0f;
	struct timeval start, end;

	gettimeofday(&start, NULL);
	gather_mul<<<grid, block>>>(d_source, d_res, r_len, d_loc,from,to);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}

double gatherImpl_mul(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize) {
	Record *d_source, *d_res;
	int *d_loc;
	double totalTime = 0.0f;

	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_loc,sizeof(int)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_loc, h_loc, sizeof(int) * r_len, cudaMemcpyHostToDevice);

	int numRun=8;
	if(r_len<256*1024)
		numRun=1;
	else
		if(r_len<1024*1024)
			numRun=2;
		else
			if(r_len<8192*1024)
				numRun=4;
	printf("run: %d\n", numRun);

	int runSize=r_len/numRun;	
	if(r_len%numRun!=0)
		runSize+=1;

	for(int i = 0; i < numRun; i++) {
		int from = i * runSize;
		int to = (i+1) * runSize;
		totalTime += gatherDevice_mul(d_source, d_res, r_len, d_loc, blockSize, gridSize, from, to);
	}

	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));
	checkCudaErrors(cudaFree(d_loc));

	return totalTime;
}

double gatherImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize) {
	Record *d_source, *d_res;
	int *d_loc;
	double totalTime = 0.0f;

	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_loc,sizeof(int)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_loc, h_loc, sizeof(int) * r_len, cudaMemcpyHostToDevice);

	totalTime = gatherDevice(d_source, d_res, r_len, d_loc, blockSize, gridSize);
	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));
	checkCudaErrors(cudaFree(d_loc));

	return totalTime;
}