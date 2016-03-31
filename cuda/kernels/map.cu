//
//  map.cu
//  comparison_gpu/cuda
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
	const int r_len
#ifdef RECORDS
	,bool isRecord
#endif
	) 
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = gridDim.x * blockDim.x;
	
	while (threadId < r_len) {
#ifdef RECORDS
		if (isRecord)
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
	int r_len, int blockSize, int gridSize
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
	map_kernel<T><<<grid, block>>>(		
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len
#ifdef RECORDS
		,isRecord
#endif
	);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

template float map<int>(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	int *d_source_values, int *d_dest_values, 	
	int r_len, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template float map<long>(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	long *d_source_values, long *d_dest_values, 	
	int r_len, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template float map<float>(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	float *d_source_values, float *d_dest_values, 	
	int r_len, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template float map<double>(		
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	double *d_source_values, double *d_dest_values, 	
	int r_len, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);