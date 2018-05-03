#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
using namespace std;

bool testScan_thrust(int len, float& totalTime, int isExclusive) {

    bool res = true;

    //allocate for the host memory
    int *h_in_gpu = new int[len];
    int *h_in_cpu = new int[len];

    for(int i = 0; i < len; i++) {
        h_in_gpu[i] = 1;
        h_in_cpu[i] = 1;
    }

    int *d_in, *d_out;
    checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*len));

    cudaMemcpy(d_in, h_in_gpu, sizeof(int) * len, cudaMemcpyHostToDevice);

    thrust::device_ptr<int> g_ptr_in = thrust::device_pointer_cast(d_in);
    thrust::device_ptr<int> g_ptr_out = thrust::device_pointer_cast(d_out);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    thrust::exclusive_scan(g_ptr_in, g_ptr_in + len, g_ptr_out);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);

    cudaMemcpy(h_in_gpu, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

    for(int i = 0; i < len; i++) {
        if (h_in_gpu[i] != i) {
            res = false;
        }
    }

//    checkCudaErrors(cudaFree(d_in));

    delete[] h_in_gpu;
    delete[] h_in_cpu;

    return res;
}

int main() {
    float idnElapsedTime;
    int dataSize = 16000000;
    bool res = testScan_thrust(dataSize, idnElapsedTime, 1);
    if (res)    cout<<"right"<<endl;
    else
        cout<<"wrong"<<endl;
    cout<<"data size:"<<dataSize* sizeof(int)/1024/1024<<" MB"<<endl;
    cout<<"total time:"<<idnElapsedTime<<" ms"<<endl;
    cout<<"throughput: "<<dataSize* sizeof(int)/idnElapsedTime/1e6<<" GB/s"<<endl;

    return 0;
}