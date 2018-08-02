//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Plat.h"
using namespace std;

int main(int argc, const char *argv[]) {
    Plat::plat_init();

//    device_param_t param = Plat::get_device_param();
//    cout<<"context"<<param.context<<endl;
//    cout<<"gmem size:"<<param.gmem_size*1.0/1024/1024/1024<<"GB"<<endl;
//    cout<<"lmem size:"<<param.lmem_size<<endl;
//    cout<<"Cache line size:"<<param.cacheline_size<<endl;
//    cout<<"Compute units:"<<param.cus<<endl;
//    cout<<"Maximal local size:"<<param.max_local_size<<endl;
//    cout<<"Max Object alloc size:"<<param.max_alloc_size*1.0/1024/1024/1024<<"GB"<<endl;

//      testMem(info);
//      testAccess(info);
//      testLatency(info);

    //test the gather and scatter with uniformly distributed indexes
//    for(int i = 128; i < 4096; i += 256) {
//        int num = i / sizeof(int) * 1024 * 1024;
//        testGather(num, info);
//        cout<<endl;
//    }
//
//    for(int i = 128; i < 4096; i += 256) {
//        int num = i / sizeof(int) * 1024 * 1024;
//        testScatter(num, info);
//        cout<<endl;
//    }


//    testAtomic(info);

//        testScatter(num, info);

    //test scan
//    for(int scale = 10; scale <= 30; scale++) {
//        int num = 1<<scale;
//        double aveTime;
//        cout<<scale<<'\t';
////        bool res = testScan(num, aveTime, 64, 39, 112, 0);    //CPU testing
////        bool res = testScan(num, aveTime, 64, 240, 67, 0);    //MIC testing
//        bool res = testScan(num, aveTime, 1024, 15, 0, 11, false);    //GPU testing
//        if (!res) {
//            cerr<<"Wrong result!"<<endl;
//            exit(1);
//        }
//        cout<<"Time:"<<aveTime<<" ms.\t";
//        cout<<"Throughput:"<<num*1.0 /1024/1024/1024/aveTime*1000<<" GKeys/s"<<endl;
//    }

//    uint64_t length = 1<<29; //2GB
//    test_wg_sequence(length);



//     testScanParameters(length, 1, info);
//    double totalTime;
//    int length = 1<<25;
//    bool res = testScan(length, totalTime, 1024, 15, 0, 11, false);    //gpu
//    bool res = testScan(length, totalTime, 64, 39, 112, 0, false);    //cpu
//    bool res = testScan(length, totalTime, 64, 240, 33, 1, info);    //mic

//     if (res)   cout<<"right ";
//     else       cout<<"wrong ";
//
//     cout<<"Time:"<<totalTime<<" ms.\t";
//     cout<<"Throughput:"<<length*1.0/1024/1024/1024/totalTime*1000<<" GKeys/s"<<endl;


//    for (int buckets = 2; buckets <= 4096; buckets <<= 1) {
//        split_test_parameters(length, buckets, WG_reorder, KVS_AOS, 2, info);
//    }

    int length = 1<<25;
    cout<<"Key-values:"<<endl;
    for(int buckets = 512; buckets <= 512; buckets<<=1) {
        split_test_specific(
                length, buckets,
                WG_fixed_reorder, KO,
                1, 39);
    }

//------- finished operations ---------------


    // runMem<double>();
    // runAtomic();
    // runBarrier(experTime);

     // runMap();
     // testGather(fixedValues, 1000000000, info);
//     testScatter(fixedValues, 1000000000, info);
    // radixSortTime = runRadixSort(experTime);

//    scanTime = runScan(experTime, scan_blockSize);
//    cout<<"Time for scan: "<<scanTime<<" ms."<<'\t'<<"BlockSize: "<<scan_blockSize<<endl;
    // cout<<"Time for radix sort: "<<radixSortTime<<" ms."<<endl;


//    testBitonitSort(fixedRecords, dataSize, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, dataSize, info, 0, totalTime);      //0:  descending

//test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    int num = 1600000;
//    testHj(dataSize, dataSize, info);         //16: lower 16 bits to generate the buckets

    return 0;
}
