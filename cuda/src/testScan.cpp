#include "test.h"

using namespace std;

template<typename T>
bool testScan_warp(T *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize) {
	
	bool res = true;
	
	//allocate for the host memory
	T *h_source_gpu = new T[r_len];
	T *h_source_cpu = new T[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source_gpu[i] = source[i];
		h_source_cpu[i] = source[i];
	}
	
	T *d_source;
    checkCudaErrors(cudaMalloc(&d_source,sizeof(T)*r_len));

    // int *h_source_thrust = new int[r_len];          //host memory
    // for(int i = 0; i < r_len; i++) h_source_thrust[i] = h_source[i];

    cudaMemcpy(d_source, h_source_gpu, sizeof(T) * r_len, cudaMemcpyHostToDevice);
    totalTime = scan_warpwise<T>(d_source, r_len, isExclusive, blockSize);
    cudaMemcpy(h_source_gpu, d_source, sizeof(T) * r_len, cudaMemcpyDeviceToHost);

	// totalTime = scanImpl(h_source_gpu, r_len, blockSize, gridSize, isExclusive);

	// checking 
	if (isExclusive == 0) {         //inclusive
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = source[i] + h_source_cpu[i-1];
        }
    }
    else {                          //exclusive
        h_source_cpu[0] = 0;
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = h_source_cpu[i-1] + source[i-1];
        }
    }
    
    for(int i = 0; i < r_len; i++) {
        if (h_source_cpu[i] != h_source_gpu[i]) res = false;
    }

    checkCudaErrors(cudaFree(d_source));

	delete[] h_source_gpu;
	delete[] h_source_cpu;
	
	return res;
}

//templates
template bool testScan_warp<int>(int *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_warp<long>(long *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_warp<float>(float *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_warp<double>(double *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);

template<typename T>
bool testScan_ble(T *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize) {
    
    bool res = true;
    
    //allocate for the host memory
    T *h_source_gpu = new T[r_len];
    T *h_source_cpu = new T[r_len];

    for(int i = 0; i < r_len; i++) {
        h_source_gpu[i] = source[i];
        h_source_cpu[i] = source[i];
    }
    
    T *d_source;
    checkCudaErrors(cudaMalloc(&d_source,sizeof(T)*r_len));

    // int *h_source_thrust = new int[r_len];          //host memory
    // for(int i = 0; i < r_len; i++) h_source_thrust[i] = h_source[i];

    cudaMemcpy(d_source, h_source_gpu, sizeof(T) * r_len, cudaMemcpyHostToDevice);
    totalTime = scan_ble<T>(d_source, r_len, isExclusive, blockSize);
    cudaMemcpy(h_source_gpu, d_source, sizeof(T) * r_len, cudaMemcpyDeviceToHost);

    // totalTime = scanImpl(h_source_gpu, r_len, blockSize, gridSize, isExclusive);

    // checking 
    if (isExclusive == 0) {         //inclusive
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = source[i] + h_source_cpu[i-1];
        }
    }
    else {                          //exclusive
        h_source_cpu[0] = 0;
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = h_source_cpu[i-1] + source[i-1];
        }
    }
    
    for(int i = 0; i < r_len; i++) {
        if (h_source_cpu[i] != h_source_gpu[i]) res = false;
    }

    checkCudaErrors(cudaFree(d_source));

    delete[] h_source_gpu;
    delete[] h_source_cpu;
    
    return res;
}

template bool testScan_ble<int>(int *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_ble<long>(long *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_ble<float>(float *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);
template bool testScan_ble<double>(double *source, int r_len, float& totalTime, int isExclusive,  int blockSize, int gridSize);

