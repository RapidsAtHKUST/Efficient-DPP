#include "test.h"

template<class T>
bool testMap(T *source, int r_len, float& totalTime, int blockSize, int gridSize) {
	
	bool res = true;

	//allocate for the host memory
	T *h_source = new T[r_len];
	T *h_res = new T[r_len];
	T *d_source, *d_res;

	for(int i = 0; i < r_len; i++) {
		h_source[i] = source[i];
	}
	
	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(T)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(T)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(T) * r_len, cudaMemcpyHostToDevice);
	totalTime = map<T>(d_source, d_res, r_len, blockSize, gridSize);
	cudaMemcpy(h_res, d_res, sizeof(T)*r_len, cudaMemcpyDeviceToHost);	
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));

	//checking 
	for(int i = 0; i < r_len; i++) {
#ifdef RECORDS
		if (h_source[i].x != h_res[i].x ||
			floorOfPower2_CPU(h_source[i].y) != h_res[i].y) 
#else
		if (h_res[i] != floorOfPower2_CPU(h_source[i]))
#endif
		{
				res = false;
				break;
		}
	}

	delete[] h_source;
	delete[] h_res;

	return res;
}

#ifdef RECORDS
	template bool testMap<Record>(Record *source, int r_len, float& totalTime, int blockSize, int gridSize);
#else
	template bool testMap<int>(int *source, int r_len, float& totalTime, int blockSize, int gridSize);
#endif

