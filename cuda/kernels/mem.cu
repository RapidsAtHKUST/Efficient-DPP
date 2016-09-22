//
//  map.cu
//  comparison_gpu/cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"

#define COALESCED_ADD(time)       (v += d_source_values[globalId + time]); 

__global__ void mem_read( 
	int *d_source_values, int *d_dest_values) 
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int v = 0;
	int global_output = globalId;
	COALESCED_ADD( 0 ); COALESCED_ADD( 8388608 ); COALESCED_ADD( 16777216 );COALESCED_ADD( 25165824 );
    COALESCED_ADD( 33554432 ); COALESCED_ADD( 41943040 ); COALESCED_ADD( 50331648 );COALESCED_ADD( 58720256 );
    COALESCED_ADD( 67108864 ); COALESCED_ADD( 75497472 ); COALESCED_ADD( 83886080 );COALESCED_ADD( 92274688 );
    COALESCED_ADD( 100663296 ); COALESCED_ADD( 109051904 ); COALESCED_ADD( 117440512 );COALESCED_ADD( 125829120 );
    COALESCED_ADD( 134217728 ); COALESCED_ADD( 142606336 ); COALESCED_ADD( 150994944 );COALESCED_ADD( 159383552 );
    COALESCED_ADD( 167772160 ); COALESCED_ADD( 176160768 ); COALESCED_ADD( 184549376 );COALESCED_ADD( 192937984 );
    COALESCED_ADD( 201326592 ); COALESCED_ADD( 209715200 ); COALESCED_ADD( 218103808 );COALESCED_ADD( 226492416 );
    COALESCED_ADD( 234881024 ); COALESCED_ADD( 243269632 ); COALESCED_ADD( 251658240 );COALESCED_ADD( 260046848 );
    COALESCED_ADD( 268435456 ); COALESCED_ADD( 276824064 ); COALESCED_ADD( 285212672 );COALESCED_ADD( 293601280 );
    COALESCED_ADD( 301989888 ); COALESCED_ADD( 310378496 ); COALESCED_ADD( 318767104 );COALESCED_ADD( 327155712 );
    COALESCED_ADD( 335544320 ); COALESCED_ADD( 343932928 ); COALESCED_ADD( 352321536 );COALESCED_ADD( 360710144 );
    COALESCED_ADD( 369098752 ); COALESCED_ADD( 377487360 ); COALESCED_ADD( 385875968 );COALESCED_ADD( 394264576 );
    COALESCED_ADD( 402653184 ); COALESCED_ADD( 411041792 ); COALESCED_ADD( 419430400 );COALESCED_ADD( 427819008 );
    COALESCED_ADD( 436207616 ); COALESCED_ADD( 444596224 ); COALESCED_ADD( 452984832 );COALESCED_ADD( 461373440 );
    COALESCED_ADD( 469762048 ); COALESCED_ADD( 478150656 ); COALESCED_ADD( 486539264 );COALESCED_ADD( 494927872 );
    COALESCED_ADD( 503316480 ); COALESCED_ADD( 511705088 ); COALESCED_ADD( 520093696 );COALESCED_ADD( 528482304 );
    COALESCED_ADD( 536870912 ); COALESCED_ADD( 545259520 ); COALESCED_ADD( 553648128 );COALESCED_ADD( 562036736 );
    COALESCED_ADD( 570425344 ); COALESCED_ADD( 578813952 ); COALESCED_ADD( 587202560 );COALESCED_ADD( 595591168 );
    COALESCED_ADD( 603979776 ); COALESCED_ADD( 612368384 ); COALESCED_ADD( 620756992 );COALESCED_ADD( 629145600 );
    COALESCED_ADD( 637534208 ); COALESCED_ADD( 645922816 ); COALESCED_ADD( 654311424 );COALESCED_ADD( 662700032 );

    d_dest_values[global_output] = v;
}

__global__ void mem_write(int *d_dest_values) 
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    d_dest_values[globalId] = 12345;
}

__global__ void mem_mul(
	int2 *d_source_values,
	int2 *d_dest_values) 
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    d_dest_values[globalId].x = d_source_values[globalId].x * 3;
    d_dest_values[globalId].y = d_source_values[globalId].y * 3;

}

float testMemRead(int *d_source_values, int *d_dest_values, int blockSize, int gridSize) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	mem_read<<<grid, block>>>(d_source_values, d_dest_values);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

float testMemWrite(int *d_dest_values, int blockSize, int gridSize) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	mem_write<<<grid, block>>>(d_dest_values);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

float testMemMul(int2 *d_source_values, int2 *d_dest_values, int blockSize, int gridSize) 
{
	dim3 grid(gridSize);
	dim3 block(blockSize);

	float totalTime = 0.0f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	mem_mul<<<grid, block>>>(d_source_values, d_dest_values);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&totalTime, start, end);

	return totalTime;
}

