#include "kernels.h"

__device__
int floorOfPower2(int a) {
	int base = 1;
	while (base < a) {
		base <<= 1;
	}
	return base>>1;
}

__global__
void map(Record *d_source, Record *d_res, int r_len) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = gridDim.x * blockDim.x;
	
	while (threadId < r_len) {
		d_res[threadId].x = d_source[threadId].x;
		d_res[threadId].y = floorOfPower2(d_source[threadId].y);
		threadId += threadNum;
	}
}

void mapImpl(Record *h_source, Record *h_res, int r_len) {
	Record *d_source, *d_res;

	dim3 grid(256);
	dim3 block(512);
	
	//allocate for the device memory
	checkCudaErrors(cudaMalloc(&d_source,sizeof(Record)*r_len));
	checkCudaErrors(cudaMalloc(&d_res,sizeof(Record)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(Record) * r_len, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	map<<<grid, block>>>(d_source, d_res, r_len);
	cudaDeviceSynchronize();

	cudaMemcpy(h_res, d_res, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);	

	//cout<<"ints:"<<endl;
	//for(int i = 0; i < r_len; i++) {
	//	cout<<h_source[i].x<<' '<<h_source[i].y<<'\t'<<h_res[i].x <<' '<<h_res[i].y<<endl;
	//}
	
	checkCudaErrors(cudaFree(d_res));
	checkCudaErrors(cudaFree(d_source));
}





