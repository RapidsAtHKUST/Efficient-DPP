//
//  radixSort.cu
//  comparison_gpu
//
//  Created by Bryan on 01/26/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"
#define BITS 4
#define RADIX (1<<BITS)

__global__
void countHis(const Record* source,
              const int length,
			  int* histogram,        //size: globalSize * RADIX
              const int shiftBits)
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = ceil(1.0*length / globalSize);
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    //initialization
    for(int i = 0; i < RADIX; i++) {
        temp[i + offset] = 0;
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[id].y;
        current = (current >> shiftBits) & mask;
        temp[offset + current]++;
    }
    __syncthreads();
    
    for(int i = 0; i < RADIX; i++) {
        histogram[i*globalSize + globalId] = temp[offset+i];
    }
}

__global__
void writeHis(const Record* source,
			  const int length,
              const int* histogram,
              int* loc,              //size equal to the size of source
              const int shiftBits)               
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = ceil(1.0 *length / globalSize);     // length for each thread to proceed
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    for(int i = 0; i < RADIX; i++) {
        temp[offset + i] = histogram[i*globalSize + globalId];
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[globalId * elePerThread + i].y;
        current = (current >> shiftBits) & mask;
        loc[globalId * elePerThread + i] = temp[offset + current];
        temp[offset + current]++;
    }
}

__global__
void countHis_int(const int* source,
              	  const int length,
			  	  int* histogram,        //size: globalSize * RADIX
              	  const int shiftBits)
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = ceil(1.0*length / globalSize);
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    //initialization
    for(int i = 0; i < RADIX; i++) {
        temp[i + offset] = 0;
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[id];
        current = (current >> shiftBits) & mask;
        temp[offset + current]++;
    }
    __syncthreads();
    
    for(int i = 0; i < RADIX; i++) {
        histogram[i*globalSize + globalId] = temp[offset+i];
    }
}

__global__
void writeHis_int(const int* source,
			  const int length,
              const int* histogram,
              int* loc,              //size equal to the size of source
              const int shiftBits)               
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = ceil(1.0 *length / globalSize);     // length for each thread to proceed
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    for(int i = 0; i < RADIX; i++) {
        temp[offset + i] = histogram[i*globalSize + globalId];
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[globalId * elePerThread + i];
        current = (current >> shiftBits) & mask;
        loc[globalId * elePerThread + i] = temp[offset + current];
        temp[offset + current]++;
    }
}

double radixSortDevice(Record *d_source, int r_len, int blockSize, int gridSize) {
	blockSize = 512;
	gridSize = 256;

	double totalTime = 0.0f;
	int globalSize = blockSize * gridSize;

	//histogram
	int *his, *loc, *res_his;
	checkCudaErrors(cudaMalloc(&his, sizeof(int)*globalSize * RADIX));
	checkCudaErrors(cudaMalloc(&loc, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&res_his, sizeof(int)*globalSize * RADIX));

	Record *d_temp;
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(Record)*r_len));

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	thrust::device_ptr<int> dev_his(his);
	thrust::device_ptr<int> dev_res_his(res_his);

	gettimeofday(&start,NULL);
	for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {
		countHis<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
		scanDevice(his, globalSize*RADIX, 1024, 1024,1);
		writeHis<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
		scatterDevice(d_source,d_temp, r_len, loc, 1024,32768);
		cudaMemcpy(d_source, d_temp, sizeof(Record)*r_len, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);
	totalTime = diffTime(end, start);

	checkCudaErrors(cudaFree(his));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(loc));

	return totalTime;
}

double radixSortDevice_int(int *d_source, int r_len, int blockSize, int gridSize) {
	blockSize = 512;
	gridSize = 256;

	double totalTime = 0.0f;
	int globalSize = blockSize * gridSize;

	//histogram
	int *his, *loc, *res_his;
	checkCudaErrors(cudaMalloc(&his, sizeof(int)*globalSize * RADIX));
	checkCudaErrors(cudaMalloc(&loc, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&res_his, sizeof(int)*globalSize * RADIX));

	int *d_temp;
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(int)*r_len));

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	thrust::device_ptr<int> dev_his(his);
	thrust::device_ptr<int> dev_res_his(res_his);

	gettimeofday(&start,NULL);
	for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {
		countHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
		scanDevice(his, globalSize*RADIX, 1024, 1024,1);
		writeHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
		scatterDevice_int(d_source,d_temp, r_len, loc, 1024,32768);
		cudaMemcpy(d_source, d_temp, sizeof(int)*r_len, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);
	totalTime = diffTime(end, start);

	checkCudaErrors(cudaFree(his));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(loc));

	return totalTime;
}

double radixSortImpl(Record *h_source, int r_len, int blockSize, int gridSize) {
	double totalTime = 0.0f;
	Record *d_source;
	
	//thrust test
	int *keys = new int[r_len];
	int *values = new int[r_len];

	for(int i = 0; i < r_len; i++) {
		keys[i] = h_source[i].x;
		values[i] = h_source[i].y;
	}

	checkCudaErrors(cudaMalloc(&d_source, sizeof(Record)*r_len));
	cudaMemcpy(d_source, h_source, sizeof(Record)*r_len, cudaMemcpyHostToDevice);

	totalTime = radixSortDevice(d_source, r_len, blockSize, gridSize);

	cudaMemcpy(h_source, d_source, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(d_source));


	// struct timeval start, end;

	// gettimeofday(&start, NULL);
	// thrust::sorting::stable_radix_sort_by_key(values, values+r_len, keys);
	// gettimeofday(&end, NULL);

	// for(int i = 0; i < r_len; i++) {
	// 	cout<<h_source[i].x<<' '<<h_source[i].y<<'\t'<<keys[i]<<' '<<values[i]<<endl;
	// }
	// double thrustTime = diff(end,start);
	// cout<<"Thrust time for radixsort: "<<thrustTime<<" ms."<<endl;

	delete[] keys;
	delete[] values;
	return totalTime;
}

double radixSortImpl_int(int *h_source, int r_len, int blockSize, int gridSize) {
	double totalTime = 0.0f;
	int *h_thrust_source = new int[r_len];
	
	int *d_source;
	int *d_thrust_source;

	checkCudaErrors(cudaMalloc(&d_source, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&d_thrust_source, sizeof(int)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(int)*r_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_thrust_source, h_source, sizeof(int)*r_len, cudaMemcpyHostToDevice);

	totalTime = radixSortDevice_int(d_source, r_len, blockSize, gridSize);

	cudaMemcpy(h_source, d_source, sizeof(int)*r_len, cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(d_source));

	struct timeval start, end;

	thrust::device_ptr<int> dev_source(d_thrust_source);

	gettimeofday(&start, NULL);
	thrust::sort(dev_source, dev_source+r_len);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	cudaMemcpy(h_thrust_source, d_thrust_source, sizeof(int) * r_len, cudaMemcpyDeviceToHost);
	
	double thrustTime = diffTime(end,start);
	std::cout<<"Thrust time for radixsort: "<<thrustTime<<" ms."<<std::endl;
	
	//check the thrust output with implemented output
	for(int i = 0; i < r_len; i++) {
		if (h_source[i] != h_thrust_source[i])	std::cerr<<"different"<<std::endl;
	}
	delete[] h_thrust_source;
	return totalTime;
}