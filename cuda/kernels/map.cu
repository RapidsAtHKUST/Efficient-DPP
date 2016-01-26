//
//  map.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

__device__
int floorOfPower2(int a) {
	int base = 1;
	while (base < a) {
		base <<= 1;
	}
	return base>>1;
}

__global__
void map(Record *d_source, Record *d_res, int r_len) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = gridDim.x * blockDim.x;
	
	while (threadId < r_len) {
		d_res[threadId].x = d_source[threadId].x;
		d_res[threadId].y = floorOfPower2(d_source[threadId].y);
		threadId += threadNum;
	}
}

double mapDevice(Record *d_source, Record *d_res, int r_len, int blockSize, int gridSize) {

	dim3 grid(gridSize);
	dim3 block(blockSize);

	double totalTime = 0.0f;
	struct timeval start, end;

	gettimeofday(&start, NULL);
	cudaDeviceSynchronize();
	map<<<grid, block>>>(d_source, d_res, r_len);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime = diffTime(end, start);

	return totalTime;
}


double mapImpl(Record *h_source, Record *h_res, int r_len, int blockSize, int gridSize) {

	double totalTime = 0.0f;
	
	Record *d_source, *d_res;
	
	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);
	totalTime = mapDevice(d_source, d_res, r_len, blockSize, gridSize);
	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));

	return totalTime;
}





