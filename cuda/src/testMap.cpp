#include "test.h"

template<class T>
bool testMap( 
#ifdef RECORDS
	int *source_keys, 
#endif
	T *source_values, int r_len, 
	float& totalTime, int blockSize, int gridSize) {
	
	bool res = true;

	//allocate for the host memory
#ifdef RECORDS
	T *h_source_keys = new T[r_len];
	T *h_dest_keys = new T[r_len];
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
	T *d_source_keys, *d_dest_keys;
	checkCudaErrors(cudaMalloc(&d_source_keys,sizeof(T)*r_len));
	checkCudaErrors(cudaMalloc(&d_dest_keys,sizeof(T)*r_len));
	cudaMemcpy(d_source_keys, h_source_keys, sizeof(T) * r_len, cudaMemcpyHostToDevice);
#endif

	totalTime = map<T> 	
	(					
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len, blockSize, gridSize);
	cudaMemcpy(h_dest_values, d_dest_values, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
	
#ifdef RECORDS
	cudaMemcpy(h_dest_keys, d_dest_keys, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
#endif
	checkCudaErrors(cudaFree(d_dest_values));
	checkCudaErrors(cudaFree(d_source_values));
#ifdef RECORDS
	checkCudaErrors(cudaFree(d_dest_keys));
	checkCudaErrors(cudaFree(d_source_keys));
#endif

	//checking 
	for(int i = 0; i < r_len; i++) {

		if (floorOfPower2_CPU(h_source_values[i]) != h_dest_values[i]	
#ifdef RECORDS
		|| h_source_keys[i] != h_dest_keys[i]
#endif
		)
		{
				res = false;
				break;
		}
	}

	delete[] h_source_values;
	delete[] h_dest_values;

	return res;
}

template
bool testMap<int>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	int *source_values, int r_len, 
	float& totalTime, int blockSize, int gridSize);

