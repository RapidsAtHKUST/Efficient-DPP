/*
 * compile: nvcc -o scanThrust -O3 scanThrust.cu -I /usr/local/cuda/samples/common/inc/
 */
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
using namespace std;

bool testScan_thrust(int len, int isExclusive) {

    bool res = true;
    float totalTime = 0.0f;

    int experTime = 10;
    for(int e = 0; e < experTime; e++) {
        //allocate for the host memory
        float tempTime;
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
        cudaEventElapsedTime(&tempTime, start, end);

        cudaMemcpy(h_in_gpu, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

        if(e==0) {      //check
            for(int i = 0; i < len; i++) {
                if (h_in_gpu[i] != i) {
                    res = false;
                }
            }
        }
        else if (res == true) {
            totalTime += tempTime;
        }
        else {
            cout<<"wrong"<<endl;
            break;
        }

        checkCudaErrors(cudaFree(d_in));
        checkCudaErrors(cudaFree(d_out));

        delete[] h_in_gpu;
        delete[] h_in_cpu;
    }
    totalTime/= (experTime-1);

    cout<<"Data num:" <<len<<'\t'
        <<"Time:"<<totalTime<<" ms"<<'\t'
        <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/totalTime*1e3<<" GB/s"<<endl;

    return res;
}

int main() {
    for(int scale = 10; scale <= 30; scale++) {
        int num = pow(2,scale);
        cout<<scale<<'\t';
        testScan_thrust(num, 1);
    }

    return 0;
}