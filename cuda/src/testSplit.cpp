#include "test.h"
using namespace std;

template<class T> bool testSplit(
#ifdef RECORDS
	int *source_keys, 
#endif
	T *source_values,int r_len, float& totalTime,  
	int fanout, int blockSize, int gridSize) 
{

	bool res = true;
	
#ifdef RECORDS
	int *h_source_keys = new int[r_len];
	int *h_dest_keys = new int[r_len];
#endif
	T *h_source_values = new T[r_len];
	T *h_dest_values = new T[r_len];

	T *d_source_values, *d_dest_values;

	for(int i = 0; i < r_len; i++) {
		h_source_values[i] = source_values[i];
#ifdef RECORDS
		h_source_keys[i] = source_keys[i];
#endif
	}
	
	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source_values,sizeof(T)*r_len));
	checkCudaErrors(cudaMalloc(&d_dest_values,sizeof(T)*r_len));
	cudaMemcpy(d_source_values, h_source_values, sizeof(T) * r_len, cudaMemcpyHostToDevice);

#ifdef RECORDS
	int *d_source_keys, *d_dest_keys;
	checkCudaErrors(cudaMalloc(&d_source_keys,sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&d_dest_keys,sizeof(int)*r_len));
	cudaMemcpy(d_source_keys, h_source_keys, sizeof(int) * r_len, cudaMemcpyHostToDevice);
#endif

	int *d_his;
	int globalSize = blockSize * gridSize;
	checkCudaErrors(cudaMalloc(&d_his,sizeof(int)* globalSize * fanout));

	totalTime = split<T> 	
	(					
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, d_his, r_len, fanout, blockSize, gridSize
#ifdef RECORDS
	,true
#endif
		);
	cudaMemcpy(h_dest_values, d_dest_values, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
	
#ifdef RECORDS
	cudaMemcpy(h_dest_keys, d_dest_keys, sizeof(int)*r_len, cudaMemcpyDeviceToHost);	
#endif
	checkCudaErrors(cudaFree(d_dest_values));
	checkCudaErrors(cudaFree(d_source_values));
	checkCudaErrors(cudaFree(d_his));

#ifdef RECORDS
	checkCudaErrors(cudaFree(d_dest_keys));
	checkCudaErrors(cudaFree(d_source_keys));
#endif
	
	//checking
    for(int i = 1; i < r_len; i++) {
        if (h_dest_values[i] < h_dest_values[i-1])  {
        	res = false;
        	break;
        }
    }

	delete[] h_source_values;
	delete[] h_dest_values;
#ifdef RECORDS
	delete[] h_source_keys;
	delete[] h_dest_keys;
#endif

    return res;
}

//template
template bool testSplit<int>(
#ifdef RECORDS
	int *source_keys, 
#endif
	int *source_values,int r_len, float& totalTime,  
	int fanout, int blockSize, int gridSize) ;