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

	while (base < (int)a) {
		base <<= 1;
	}
	return base>>1;
}


template<class T>
__global__ void map_kernel( 
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys, 
#endif
	T *d_source_values, T *d_dest_values,
	int r_len) 
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = gridDim.x * blockDim.x;
	
	while (threadId < r_len) {
#ifdef RECORDS
		d_dest_keys[threadId] = d_source_keys[threadId];
#endif
		d_dest_values[threadId] = floorOfPower2<T>(d_source_values[threadId]);
		threadId += threadNum;
	}
}


template<class T>
float map(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int blockSize, int gridSize) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	map_kernel<T><<<grid, block>>>(		
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

template
float map<int>(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	int *d_source_values, int *d_dest_values, 	
	int r_len, int blockSize, int gridSize);


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





