/*
 * compile: nvcc -o gather -arch=sm_35 -O3 gather.cu -I /usr/local/cuda/samples/common/inc/
 */
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
using namespace std;

#define MAX_SHUFFLE_TIME (2099999999)

__global__ void gather_kernel(int *d_in, int *d_out, int *d_idx, int num)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalSIze = blockDim.x * gridDim.x;

    while (globalId < num) {
        d_out[d_idx[globalId]] = d_in[globalId];
        globalId += globalSIze;
    }
}

float scatter(int *d_in, int *d_out, int *d_idx, int num)
{
    int blockSize = 1024;
    int ele_per_thread = 16;
    int gridSize = num / blockSize / ele_per_thread;
    dim3 grid(gridSize);
    dim3 block(blockSize);

    float totalTime;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    gather_kernel<<<grid, block>>>(d_in, d_out, d_idx, num);
    cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout<<cudaGetErrorString(err)<<std::endl;

    return totalTime;
}

void test_bandwidth(int len) {
    std::cout<<"Data size(Copy): "<<len<<" ("<<len/1024/1024*sizeof(int)<<"MB)"<<'\t';

    float aveTime = 0.0;

    int *h_in, *d_in, *d_out, *h_idx, *d_idx;
    h_in = new int[len];
    h_idx = new int[len];
    for(int i = 0; i < len; i++) {
        h_in[i] = i;
        h_idx[i] = i;
    }
    unsigned shuffleTime = (len * 3 < MAX_SHUFFLE_TIME)? len*3 : MAX_SHUFFLE_TIME;

    srand((unsigned)time(NULL));
    sleep(1);

    /*data shuffling*/
    int temp, from = 0, to = 0;
    for(int i = 0; i < shuffleTime; i++) {
        from = rand() % len;
        to = rand() % len;
        temp = h_idx[from];
        h_idx[from] = h_idx[to];
        h_idx[to] = temp;
    }

    checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_idx,sizeof(int)*len));

    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, sizeof(int)*len, cudaMemcpyHostToDevice);

    int experTime = 10;
    for(int i = 0; i < experTime; i++) {
        float tempTime = scatter(d_in, d_out, d_idx, len);
        if (i != 0)     aveTime += tempTime;
    }
    aveTime /= (experTime - 1);

    delete[] h_in;
    delete[] h_idx;
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_idx));
    checkCudaErrors(cudaFree(d_out));

    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl; //compared with scan
}

int main()
{
    /*MB*/
    for(int data_size_MB = 128; data_size_MB < 4096; data_size_MB += 256) {
        int data_size = data_size_MB/ sizeof(int) * 1024 * 1024;
        test_bandwidth(data_size);
    }

    return 0;
}