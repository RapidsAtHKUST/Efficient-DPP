//
//  map.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

//mapping function 1:

template<class T>
__device__ T floorOfPower2(T a) {
	int base = 1;
#ifdef RECORDS
	int b = a.y;
#else
	int b = a; 
#endif
	while (base < b) {
		base <<= 1;
	}
#ifdef RECORDS
	T res;
	res.x = a.x;
	res.y = base>>1;
	return res;
#else
	return base>>1;
#endif
}


template<class T>
__global__ void map_kernel(T *d_source, T *d_res, int r_len) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = gridDim.x * blockDim.x;
	
	while (threadId < r_len) {
		d_res[threadId] = floorOfPower2<T>(d_source[threadId]);
		threadId += threadNum;
	}
}


template<class T>
float map(T *d_source, T  *d_res, int r_len, int blockSize, int gridSize) {

	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	map_kernel<T><<<grid, block>>>(d_source, d_res, r_len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

#ifdef RECORDS
	template float map<Record>(Record *d_source, Record *d_res, int r_len, int blockSize, int gridSize);
#else
	template float map<int>(int *d_source, int  *d_res, int r_len, int blockSize, int gridSize);
#endif


// double mapImpl(Record *h_source, Record *h_res, int r_len, int blockSize, int gridSize) {

// 	double totalTime = 0.0f;
	
// 	Record *d_source, *d_res;
	
// 	//allocate for the device memory
// 	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
// 	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));

// 	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);
// 	totalTime = map<Record>(d_source, d_res, r_len, blockSize, gridSize);
// 	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	
	
// 	checkCudaErrors(cudaFree(d_res));
// 	checkCudaErrors(cudaFree(d_source));

// 	return totalTime;
// }





