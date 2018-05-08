/*
 * compile: nvcc -o scanCUB -O3 scanCUB.cu -I /usr/local/cuda/samples/common/inc/ -I.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cub/cub/device/device_scan.cuh>
using namespace std;
using namespace cub;

bool testScan_cub(int len, int isExclusive) {

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

        // Allocate temporary storage
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len));
        checkCudaErrors(cudaMalloc(&d_temp_storage,temp_storage_bytes));

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, len));

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
        checkCudaErrors(cudaFree(d_temp_storage));
        delete[] h_in_gpu;
        delete[] h_in_cpu;
    }
    totalTime/= (experTime-1);

    cout<<"Data num:" <<len<<'\t'
        <<"Time:"<<totalTime<<" ms"<<'\t'
        <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/bestTime*1e3<<" GB/s"<<endl;

    return res;
}

int main() {
    for(int scale = 28; scale <= 30; scale++) {
        int num = pow(2,scale);
        cout<<scale<<'\t';
        testScan_cub(num, 1);
    }

    return 0;
}