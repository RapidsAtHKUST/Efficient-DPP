// #include "test.h"

// bool testScatter(Record *source, int r_len, int *loc,double& totalTime, int blockSize, int gridSize) {
	
// 	bool res = true;

// 	//allocate for the host memory
// 	Record *h_source = new Record[r_len];
// 	Record *h_res = new Record[r_len];

// 	for(int i = 0; i < r_len; i++) {
// 		h_source[i].x = source[i].x;
// 		h_source[i].y = source[i].y;
// 	}

// 	totalTime = scatterImpl(h_source, h_res, r_len, loc,  blockSize, gridSize);

// 	//checking 
// 	for(int i = 0; i < r_len; i++) {
// 		if (h_res[loc[i]].x != h_source[i].x ||
// 			h_res[loc[i]].y != h_source[i].y) {
// 			res = false;
// 			break;
// 		}
// 	}

// 	delete[] h_source;
// 	delete[] h_res;

// 	return res;
// }
#include "test.h"

// bool testGather(Record *source, int r_len, int *loc,double& totalTime,  int blockSize, int gridSize) {
	
// 	bool res = true;

// 	//allocate for the host memory
// 	Record *h_source = new Record[r_len];
// 	Record *h_res = new Record[r_len];

// 	for(int i = 0; i < r_len; i++) {
// 		h_source[i].x = source[i].x;
// 		h_source[i].y = source[i].y;
// 	}

// 	totalTime = gatherImpl(h_source, h_res, r_len, loc,  blockSize, gridSize);

// 	//checking 
// 	for(int i = 0; i < r_len; i++) {
// 		if (h_res[i].x != h_source[loc[i]].x ||
// 			h_res[i].y != h_source[loc[i]].y) {
// 			res = false;
// 			break;
// 		}
			
// 	}

// 	delete[] h_source;
// 	delete[] h_res;

// 	return res;
// }

// bool testGather_mul(Record *source, int r_len, int *loc,double& totalTime,  int blockSize, int gridSize) {
	
// 	bool res = true;

// 	//allocate for the host memory
// 	Record *h_source = new Record[r_len];
// 	Record *h_res = new Record[r_len];

// 	for(int i = 0; i < r_len; i++) {
// 		h_source[i].x = source[i].x;
// 		h_source[i].y = source[i].y;
// 	}

// 	totalTime = gatherImpl_mul(h_source, h_res, r_len, loc,  blockSize, gridSize);

// 	//checking 
// 	for(int i = 0; i < r_len; i++) {
// 		if (h_res[i].x != h_source[loc[i]].x ||
// 			h_res[i].y != h_source[loc[i]].y) {
// 			res = false;
// 			break;
// 		}
			
// 	}

// 	delete[] h_source;
// 	delete[] h_res;

// 	return res;
// }

template<class T> bool testScatter( 
#ifdef RECORDS
	int *source_keys, 
#endif
	T *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize) {
	
	bool res = true;

	//allocate for the host memory
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

	//locations
	int *d_loc;
	checkCudaErrors(cudaMalloc(&d_loc,sizeof(int)*r_len));
	cudaMemcpy(d_loc, loc, sizeof(int) * r_len, cudaMemcpyHostToDevice);

	totalTime = scatter<T> 	
	(					
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len, d_loc, blockSize, gridSize
#ifdef RECORDS
	,true
#endif
		);
	cudaMemcpy(h_dest_values, d_dest_values, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
	
#ifdef RECORDS
	cudaMemcpy(h_dest_keys, d_dest_keys, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
#endif
	checkCudaErrors(cudaFree(d_dest_values));
	checkCudaErrors(cudaFree(d_source_values));
	checkCudaErrors(cudaFree(d_loc));
#ifdef RECORDS
	checkCudaErrors(cudaFree(d_dest_keys));
	checkCudaErrors(cudaFree(d_source_keys));
#endif

	//checking 
	for(int i = 0; i < r_len; i++) {
		if (
#ifdef RECORDS
		h_dest_keys[loc[i]] != h_source_keys[i] ||
#endif
		h_dest_values[loc[i]] != h_source_values[i]) 
		{
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

template<class T> bool testScatter_mul( 
#ifdef RECORDS
	int *source_keys, 
#endif
	T *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize) 
{
	
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
	int *d_source_keys, *d_dest_keys;
	checkCudaErrors(cudaMalloc(&d_source_keys,sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&d_dest_keys,sizeof(int)*r_len));
	cudaMemcpy(d_source_keys, h_source_keys, sizeof(int) * r_len, cudaMemcpyHostToDevice);
#endif

	//locations
	int *d_loc;
	checkCudaErrors(cudaMalloc(&d_loc,sizeof(int)*r_len));
	cudaMemcpy(d_loc, loc, sizeof(int) * r_len, cudaMemcpyHostToDevice);

	totalTime = scatter_mul<T> 	
	(					
#ifdef RECORDS
		d_source_keys, d_dest_keys,
#endif
		d_source_values, d_dest_values, r_len, d_loc, blockSize, gridSize
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
	checkCudaErrors(cudaFree(d_loc));
#ifdef RECORDS
	checkCudaErrors(cudaFree(d_dest_keys));
	checkCudaErrors(cudaFree(d_source_keys));
#endif

	//checking 
	for(int i = 0; i < r_len; i++) {
		if (
#ifdef RECORDS
		h_dest_keys[loc[i]] != h_source_keys[i] ||
#endif
		h_dest_values[loc[i]] != h_source_values[i]) 
		{
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

//templates
template bool testScatter<int>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	int *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter<long>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	long *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter<float>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	float *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter<double>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	double *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);


//test gather multi-pass
template bool testScatter_mul<int>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	int *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter_mul<long>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	long *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter_mul<float>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	float *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);

template bool testScatter_mul<double>( 
#ifdef RECORDS
	int *source_keys, 
#endif
	double *source_values, int r_len, int* loc,
	float& totalTime, int blockSize, int gridSize);