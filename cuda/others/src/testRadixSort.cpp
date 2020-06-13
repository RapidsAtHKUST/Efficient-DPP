#include "test.h"
using namespace std;

//template<typename T>
//bool testRadixSort(
//#ifdef RECORDS
//    int *source_keys,
//#endif
//    T *source_values, int len, float& totalTime)
//{
//    bool res = true;
//
//    T *cpu_check = new T[len];
//
//#ifdef RECORDS
//    int *h_source_keys = new int[len];
//    int *h_dest_keys = new int[len];
//#endif
//    T *h_source_values = new T[len];
//    T *h_dest_values = new T[len];
//
//    for(int i = 0; i < len; i++) {
//#ifdef RECORDS
//    	h_source_keys[i] = source_keys[i];
//#endif
//    	h_source_values[i] = source_values[i];
//        cpu_check[i] = source_values[i];
//    }
//
//    T *d_source_values;
//    checkCudaErrors(cudaMalloc(&d_source_values, sizeof(T)* len));
//#ifdef RECORDS
//    int *d_source_keys;
//    checkCudaErrors(cudaMalloc(&d_source_keys, sizeof(int)* len));
//#endif
//
//    cudaMemcpy(d_source_values, h_source_values, sizeof(T) * len, cudaMemcpyHostToDevice);
//#ifdef RECORDS
//    cudaMemcpy(d_source_keys, h_source_keys, sizeof(int) * len, cudaMemcpyHostToDevice);
//#endif
//
//    totalTime = radixSort<T>(
//#ifdef RECORDS
//    	d_source_keys,
//#endif
//    	d_source_values, len
//#ifdef RECORDS
//        ,true
//#endif
//        );
//
//    cudaMemcpy(h_dest_values, d_source_values, sizeof(T) * len, cudaMemcpyDeviceToHost);
//#ifdef RECORDS
//    cudaMemcpy(h_dest_keys, d_source_keys, sizeof(int) * len, cudaMemcpyDeviceToHost);
//#endif
//
//    //checking
//    sort(cpu_check, cpu_check + len);
//    for(int i = 0; i < len; i++) {
//        if(h_dest_values[i] != cpu_check[i]) {
//            res = false;
//            cout<<"Failed at: "<<i<<": "<<h_dest_values[i]<<' '<<cpu_check[i]<<endl;
//            cout<<h_dest_values[i-1]<<' '<<cpu_check[i-1]<<endl;
//            cout<<h_dest_values[i+1]<<' '<<cpu_check[i+1]<<endl;
//
//            break;
//        }
//    }
//    // cout<<endl;
//
//    // thrust::device_ptr<int> dev_his(his);
//    // thrust::device_ptr<int> dev_res_his(res_his);
//
//    // gettimeofday(&start,NULL);
//
//    // for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {
//
//    //  countHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
//    //  scanDevice(his, globalSize*RADIX, 1024, 1024,1);
//    //  writeHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
//    //  scatterDevice_int(d_source,d_temp, r_len, loc, 1024,32768);
//    //  int *swapPointer = d_temp;
//    //  d_temp = d_source;
//    //  d_source = swapPointer;
//    // }
//    // cudaDeviceSynchronize();
//    // gettimeofday(&end,NULL);
//    // totalTime = diffTime(end, start);
//
//    // checkCudaErrors(cudaFree(d_source_values));
//#ifdef RECORDS
//    checkCudaErrors(cudaFree(d_source_keys));
//#endif
//
//#ifdef RECORDS
//	delete[] h_source_keys;
//    delete[] h_dest_keys;
//#endif
//    delete[] h_source_values;
//    delete[] h_dest_values;
//
//    delete[] cpu_check;
//
//    return res;
//}
//
////template
//template bool testRadixSort<int>(
//#ifdef RECORDS
//    int *source_keys,
//#endif
//    int *source_values, int len, float& totalTime) ;
//
//template bool testRadixSort<long>(
//#ifdef RECORDS
//    int *source_keys,
//#endif
//    long *source_values, int len, float& totalTime) ;