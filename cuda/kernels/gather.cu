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
		d_res[threadId].x = d_source[loc[threadId]].x;
		d_res[threadId].y = d_source[loc[threadId]].y;
		threadId += threadNum;
	}
}