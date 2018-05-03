#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <assert.h>
using namespace std;

__global__ void scan(int* lo, int length, int length_log, int *sum)
{
    int localId = threadIdx.x;
    int mask_j = (1<<(length_log-1))-1;
    int mask_k = 0;
    int temp = 1;

    int localTemp = lo[localId];
    __syncthreads();

    for(int i = 0; i < length_log; i++) {
        if (localId < (length>>1)) {            //only half of the threads execute
            int para_j = (localId >> i) & mask_j;
            int para_k = localId & mask_k;

            int j = temp - 1 + (temp<<1)*para_j;
            int k = para_k;
            lo[j+k+1] = lo[j] + lo[j+k+1];

            mask_j >>= 1;
            mask_k = (mask_k<<1)+1;
            temp <<= 1;
        }
        __syncthreads();
    }

    if (localId == length-1) *sum = lo[localId];
    lo[localId] -= localTemp;
    __syncthreads();
}

int main() {
    int bits = 10;
    int threads = 1<<bits;
    int *input = new int[threads];
    for(int i=0; i < threads;i++) input[i] = 1;

    float totalTime;

    int *d_in, *d_sum;
    cudaMalloc(&d_in, sizeof(int)*threads);
    cudaMalloc(&d_sum, sizeof(int));
    cudaMemcpy(d_in, input, sizeof(int)*threads, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int sum = 0;
    cudaEventRecord(start);
    scan<<<1,threads>>>(d_in, threads, bits, d_sum);
    cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);
    cudaMemcpy(input, d_in, sizeof(int)*threads, cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < threads;i++) {
        cout<<input[i]<<' ';
    }
    cout<<endl<<"sum:"<<sum<<endl;
    cout<<"Time: "<<totalTime<<" ms."<<endl;
    delete[] input;
    cudaFree(d_in);

}