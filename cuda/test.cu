#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
using namespace std;


__global__ void mul_kernel(int *d_in, int *d_out, int scalar)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[globalId] = d_in[globalId]*scalar;
}

float mul(int *d_in, int *d_out, int blockSize, int gridSize)
{
    int scalar = 3;
    dim3 grid(gridSize);
    dim3 block(blockSize);

    float totalTime;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    mul_kernel<<<grid, block>>>(d_in, d_out, scalar);
    cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout<<cudaGetErrorString(err)<<std::endl;

    return totalTime;
}

void testMem() {
    cudaError_t err;
    int blockSize = 1024, gridSize = 32768;
    int len = blockSize * gridSize;
    std::cout<<"Data size(Multiplication): "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    float mulTime = 0.0;

    int *h_in, *d_in, *d_out;
    h_in = new int[len];
    for(int i = 0; i < len; i++) {
        h_in[i] = i;
    }
    checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*len));
    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);

    for(int i = 0; i < 10; i++) {
        float tempTime = mul(d_in, d_out, blockSize, gridSize);

        //throw away the first result
        if (i != 0)     mulTime += tempTime;
    }
    mulTime /= (10- 1);

    delete[] h_in;
    checkCudaErrors(cudaFree(d_out));
}

int main() {
    testMem();
    return 0;
}

