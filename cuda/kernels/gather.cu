//
//  gather.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

template<typename T>
__global__ void gather_kernel(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values,
	const int r_len, const int *loc
#ifdef RECORDS
	,bool isRecord
#endif
	)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
#ifdef RECORDS
		if (isRecord)
			d_dest_keys[threadId] = d_source_keys[loc[threadId]];
#endif
		d_dest_values[threadId] = d_source_values[loc[threadId]];
		threadId += threadNum;
	}
}

template<typename T>
__global__ void gather_kernel_mul(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values,
	const int r_len, const int *loc, int from, int to
#ifdef RECORDS
	,bool isRecord
#endif
	)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;

	while (threadId < r_len) {
		int tempLoc = loc[threadId];
		if (tempLoc >= from && tempLoc < to) {
#ifdef RECORDS
		if(isRecord)
			d_dest_keys[threadId] = d_source_keys[tempLoc];
#endif
			d_dest_values[threadId] = d_source_values[tempLoc];
		}
		threadId += threadNum;
	}
}

template<class T> float gather(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	gather_kernel<T><<<grid, block>>>(
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len, loc
#ifdef RECORDS
	,isRecord
#endif
	);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

template<class T> float gather_mul(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	int numRun=8;
	if(r_len < 256*1024)			numRun=1;
	else if(r_len < 1024*1024)		numRun=2;
	else if(r_len < 8192*1024)		numRun=4;

	int runSize=r_len/numRun;	
	if(r_len%numRun != 0)			runSize+=1;

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	for(int i = 0; i < numRun; i++) {
		int from = i * runSize;
		int to = (i+1) * runSize;
		gather_kernel_mul<T><<<grid, block>>>(
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len, loc, from, to
#ifdef RECORDS
	,isRecord
#endif
		);
	}
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);
	return totalTime;
}

//one pass
template float gather<int>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	int *d_source_values, int *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather<long>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	long *d_source_values, long *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather<float>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	float *d_source_values, float *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather<double>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	double *d_source_values, double *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;


//multi-pass
template float gather_mul<int>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	int *d_source_values, int *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather_mul<long>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	long *d_source_values, long *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather_mul<float>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	float *d_source_values, float *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template float gather_mul<double>(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	double *d_source_values, double *d_dest_values, 	
	int r_len, int*loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;