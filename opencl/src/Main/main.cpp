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

    int length = 1<<25;
    int idx = 0;

//    int bsize[12] = {256,256,512,512,512,512,512,512,128,64,256,128};
//    int gsize[12] = {16384,16384,8192,8192,8192,8192,8192,8192,8192,8192,8192,16384};

    cout<<"WG_varied_reorder, KO:"<<endl;
    for (int buckets = 4096; buckets <= 4096; buckets <<= 1) {
//        split_test_parameters(length, buckets, WG, KO, 2);
        split_test_specific(length, buckets, Single_reorder, KO, 1, 32768);
    }

//    idx = 0;
//    int bsize1[12] = {512,512,512,512,512,512,64,64,128,128,64,64};
//    int gsize1[12] = {16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,32768};

//    cout<<"WG_varied_reorder, AOS:"<<endl;
//    for (int buckets = 2; buckets <= 4096; buckets <<= 1) {
////        split_test_parameters(length, buckets, WG, KO, 2);
//        split_test_specific(length, buckets, WG_varied_reorder, KVS_AOS, bsize1[idx], gsize1[idx]);
//        idx++;
//    }


//    cout<<"Key-values:"<<endl;
//    for(int buckets = 2; buckets <= 4096; buckets<<=1) {
//        split_test_specific(
//                length, buckets,
//                WG, KO,
//                128, 65536);
//    }

    return 0;
}
