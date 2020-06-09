//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Plat.h"

//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
//#if defined(__APPLE__) || defined(__MACOSX)
//#include <OpenCL/OpenCL.h>
//#else
//#include <CL-v2.1/cl.h>
//#endif
using namespace std;

int main(int argc, const char *argv[]) {

    Plat::plat_init();

    double totalTime;
    int repeat = 1;
    bool res;

//      testMem(info);
//      testAccess();
//      testLatency(info);

    //test the gather and scatter with uniformly distributed indexes
//    for(int i = 128; i < 4096; i += 256) {
//        int num = i / sizeof(int) * 1024 * 1024;
//        testGather(num);
//        cout<<endl;
//    }
    int inputPass = atoi(argv[1]);
//
    for(int i = 2048; i <= 2048; i += 256) {
        int num = i / sizeof(int) * 1024 * 1024;
        testScatter(num, inputPass);
        cout<<endl;
    }


//    testAtomic(info);

//        testScatter(num, info);

    //test scan
//    for(int scale = 10; scale <= 30; scale++) {
//        int num = 1<<scale;
//        double aveTime;
//        cout<<scale<<'\t';
////        bool res = testScan(num, aveTime, 64, 40, 112, 0, atoi(argv[1]));    //CPU testing
////        bool res = testScan(num, aveTime, 64, 240, 67, 0, atoi(argv[1]));    //MIC testing
////        bool res = testScan(num, aveTime, 256, 2048, 0, 11, false);    //GPU testing
//        int tile_size = num/240;
//        if (tile_size > 1048576)    tile_size = 1048576;
//        bool res = testScanSingleThread(num, aveTime, tile_size);
//        if (!res) {
//            cerr<<"Wrong result!"<<endl;
//            exit(1);
//        }
//        cout<<"Time:"<<aveTime<<" ms.\t";
//        cout<<"Throughput:"<<num*1.0 /1024/1024/1024/aveTime*1000<<" GKeys/s"<<endl;
//    }


//    cout<<"Serial: ";
//    res = test_scan_local_schemes(SERIAL, totalTime, repeat);
//    if (res)    cout<<"right"<<' ';
//    else        cout<<"wrong"<<' ';
//    cout<<"total time:"<<totalTime<<"ms"<<" (repeat "<<repeat<<" times)"<<endl;
//
//    cout<<"Kogge: ";
//    res = test_scan_local_schemes(KOGGE, totalTime, repeat);
//    if (res)    cout<<"right"<<' ';
//    else        cout<<"wrong"<<' ';
//    cout<<"total time:"<<totalTime<<"ms"<<" (repeat "<<repeat<<" times)"<<endl;
//
//    cout<<"Sklansky: ";
//    res = test_scan_local_schemes(SKLANSKY, totalTime, repeat);
//    if (res)    cout<<"right"<<' ';
//    else        cout<<"wrong"<<' ';
//    cout<<"total time:"<<totalTime<<"ms"<<" (repeat "<<repeat<<" times)"<<endl;
//
//    cout<<"Brent: ";
//    res = test_scan_local_schemes(BRENT, totalTime, repeat);
//    if (res)    cout<<"right"<<' ';
//    else        cout<<"wrong"<<' ';
//    cout<<"total time:"<<totalTime<<"ms"<<" (repeat "<<repeat<<" times)"<<endl;



    /*local matrix scan*/
//    for(int tile_size = 1; tile_size < 50; tile_size++) {
//        cout << "Matrix_LM: ";
//        res = test_scan_matrix(LM, totalTime, tile_size, repeat);
//        if (res) cout << "right" << ' ';
//        else cout << "wrong" << ' ';
//        cout << "total time:" << totalTime << "ms\t"
//             << " (repeat " << repeat << " times)\t"
//             << " tile:" << tile_size <<"\t"
//             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
//             << endl;
//
//        cout << "Matrix_REG: ";
//        res = test_scan_matrix(REG, totalTime, tile_size, repeat);
//        if (res) cout << "right" << ' ';
//        else cout << "wrong" << ' ';
//        cout << "total time:" << totalTime << "ms\t"
//             << " (repeat " << repeat << " times)\t"
//             << " tile:" << tile_size <<"\t"
//             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
//             << endl;
//
//        cout << "Matrix_LM_REG: ";
//        res = test_scan_matrix(LM_REG, totalTime, tile_size, repeat);
//        if (res) cout << "right" << ' ';
//        else cout << "wrong" << ' ';
//        cout << "total time:" << totalTime << "ms\t"
//             << " (repeat " << repeat << " times)\t"
//             << " tile:" << tile_size <<"\t"
//             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
//             << endl;
//
//        cout << "Matrix_LM_SERIAL: ";
//        res = test_scan_matrix(LM_SERIAL, totalTime, tile_size, repeat);
//        if (res) cout << "right" << ' ';
//        else cout << "wrong" << ' ';
//        cout << "total time:" << totalTime << "ms\t"
//             << " (repeat " << repeat << " times)\t"
//             << " tile:" << tile_size <<"\t"
//             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
//             << endl;
//    }


////    uint64_t length = 1<<29; //2GB
////    test_wg_sequence(length);



//    double totalTime;
//    int length = 1<<10;
////    bool res = testScan(length, totalTime, 1024, 15, 0, 11, false);    //gpu
//    bool res = testScan(length, totalTime, 64, 1024, 112, 0, true);    //cpu
////    bool res = testScan(length, totalTime, 64, 240, 33, 1, info);    //mic
//
//     if (res)   cout<<"right ";
//     else       cout<<"wrong ";
////
//     cout<<"Time:"<<totalTime<<" ms.\t";
//     cout<<"Throughput:"<<length*1.0/1024/1024/1024/totalTime*1000<<" GKeys/s"<<endl;

//    int length = 1<<25;
//    int idx = 0;

//    int bsize[12] = {256,256,512,512,512,512,512,512,128,64,256,128};
//    int gsize[12] = {16384,16384,8192,8192,8192,8192,8192,8192,8192,8192,8192,16384};

//    cout<<"WG_varied_reorder, KO:"<<endl;
//    for (int buckets = 2; buckets <= 4096; buckets <<= 1) {
//        split_test_parameters(length, buckets, WG, KO, 2);
//        split_test_specific(length, buckets, WG, KO, 1, 2048);
//    }

        return 0;
}
