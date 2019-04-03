/*
 * compile: nvcc -o mem_cpy -arch=sm_35 -O3 mem_cpy.cu -I /usr/local/cuda/samples/common/inc/
 */
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
using namespace std;

__global__ void mul_kernel(int *d_in, int *d_out, int num)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalSIze = blockDim.x * gridDim.x;

    while (globalId < num) {
        d_out[globalId] = d_in[globalId];
        globalId += globalSIze;
    }
}

float mul(int *d_in, int *d_out, int num)
{
    int blockSize = 1024, gridSize = 1024;
    dim3 grid(gridSize);
    dim3 block(blockSize);

    float totalTime;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    mul_kernel<<<grid, block>>>(d_in, d_out, num);
    cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout<<cudaGetErrorString(err)<<std::endl;

    return totalTime;
}

void testMem(int len) {
    std::cout<<"Data size(Copy): "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    float aveTime = 0.0;

    int *h_in, *d_in, *d_out;
    h_in = new int[len];
    for(int i = 0; i < len; i++) {
        h_in[i] = i;
    }
    checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*len));
    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);

    int experTime = 10;
    for(int i = 0; i < experTime; i++) {
        float tempTime = mul(d_in, d_out, len);
        if (i != 0)     aveTime += tempTime;
    }
    aveTime /= (experTime - 1);

    delete[] h_in;
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len/1024/1024/1024/aveTime*1e3<<" GKey/s"<<std::endl; //compared with scan

}

int main()
{
    for(int scale = 10; scale <= 30; scale++) {
        unsigned data_size = 1<<scale;
        testMem(data_size);
    }

    return 0;
}