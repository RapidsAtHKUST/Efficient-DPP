//
//  map.cu
//  comparison_gpu/cuda
//
//  Created by Zhuohang Lai on 01/19/16.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#include "kernels.h"

__global__ void copy_kernel(int *d_in, int *d_out, int scalar)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[globalId] = d_in[globalId]*scalar;
}

float mul(int *d_in, int *d_out, int blockSize, int gridSize)
{
    int scalar = 3;
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
    copy_kernel <<<grid, block>>>(d_in, d_out, scalar);
	cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout<<cudaGetErrorString(err)<<std::endl;

	return totalTime;
}


