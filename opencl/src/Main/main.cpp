//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Foundation.h"
#include <cmath>
using namespace std;

int dataSize;           //max: MAX_DATA_SIZE
int outputSizeGS;      //output size of scatter and gather 

bool is_input;          //whether to read data or fast run
PlatInfo info;          //platform configuration structure

char input_arr_dir[500];
char input_rec_dir[500];
char input_loc_dir[500];

Record *fixedRecords;

int *fixedKeys;
int *fixedValues;
int *fixedLoc;

//for basic operation testing
#define MIN_BLOCK       (128)  
#define MAX_BLOCK       (1024)
#define MIN_GRID        (256)  
#define MAX_GRID        (32768)	
#define MAX_VEC_SIZE	(16)

#define NUM_BLOCK_VAR   (4)	
#define NUM_GRID_VAR    (8)	
#define NUM_VEC_SIZE    (5)

int vec[NUM_VEC_SIZE] = {1,2,4,8,16};

//device basic operation performance matrix

//for vpu testing
Basic_info perfInfo_float[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];
Basic_info perfInfo_double[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];

//for all
Device_perf_info bestInfo;


void runBarrier(int experTime);
void runAtomic();

double runMap();
double runRadixSort(int experTime);

/*parameters:
 * executor DATASIZE
 * DATASIZE : data size
 */
int main(int argc, const char * argv[]) {

    //platform initialization
    PlatInit* myPlatform = PlatInit::getInstance(0);
    cl_command_queue queue = myPlatform->getQueue();
    cl_context context = myPlatform->getContext();
    cl_command_queue currentQueue = queue;

    info.context = context;
    info.currentQueue = currentQueue;

    if (argc != 2) {
        cerr<<"Wrong number of parameters."<<endl;
        exit(1);
    }

    // dataSize = 1000000000;
    dataSize = atoi(argv[1]);       //in MB
    assert(dataSize > 0);

    #ifdef RECORDS
//        fixedKeys = new int[dataSize];
    #endif
//        fixedValues = new int[dataSize];
    #ifdef RECORDS
//        recordRandom<int>(fixedKeys, fixedValues, dataSize);
    #else
//        valRandom<int>(fixedValues,dataSize, MAX_NUM);
    #endif

    int map_blockSize = -1, map_gridSize = -1;
    int gather_blockSize = -1, gather_gridSize = -1;
    int scatter_blockSize = -1, scatter_gridSize = -1;
    int scan_blockSize = -1;

    int length = 1<<25;
    double totalTime;

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
//      testGather(dataSize, info);
//        testScatter(num, info);

//    for(int scale = 10; scale <= 30; scale++) {
//        int num = 1<<scale;
//        double totalTime;
//        cout<<scale<<'\t';
//        testScan(num, 1, totalTime, 64, 240, 33, 1, info);
//        cout<<"Time:"<<totalTime<<" ms.\t";
//        cout<<"Throughput:"<<num*1.0* sizeof(int)/1024/1024/1024/totalTime*1000<<" GB/s"<<endl;
//    }


    // testScanParameters(length, 2, info);

//    bool res = testScan(length, 1, totalTime, 1024, 15, 0, 11, info);    //gpu
//    bool res = testScan(length, 1, totalTime, 64, 39, 121, 0, info);    //cpu
//    bool res = testScan(length, 1, totalTime, 64, 240, 33, 1, info);    //mic
//    if (res)    cout<<"right\t";
//    else        cout<<"wrong\t";
//     cout<<"Time:"<<totalTime<<" ms.\t";
//     cout<<"Throughput:"<<length*1.0* sizeof(int)/1024/1024/1024/totalTime*1000<<" GB/s"<<endl;

    testSplit(length, info, 32, totalTime);

//    for(int buckets = 2; buckets <= 4096; buckets <<= 1) {
//        testSplitParameters(length, buckets, 0, 5, info);
//    }

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

void runBarrier(int experTime) {

    cout<<"----------- Barrier test ------------"<<endl;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                //--------test map------------
                double percentage;
                testBarrier(                
                input, info,tempTime, percentage, blockSize, gridSize);

                cout<<"blockSize: "<<blockSize<<'\t'
            	<<"gridSize: "<<gridSize<<'\t'
            	<<"barrier time: "<<tempTime<<" ms\t"
            	<<"time per thread: "<<tempTime/(blockSize *  gridSize)*1e6<<" ns\t"
            	<<"percentage: "<<percentage <<"%"<<endl;
            }
        }
    }

    delete[] input;
}

double runMap() {
    int blockSize = 1024, gridSize = 8192;
    int repeat = 64, repeat_trans = 16;
    testMap(info, repeat, repeat_trans, blockSize, gridSize);
}

//no need to set blockSize and gridSize
double runRadixSort(int experTime) {
    double bestTime = MAX_TIME;
    bool res;
    
    double tempTime = MAX_TIME;
    for(int i = 0 ; i < experTime; i++) {       
        //--------test radix sort------------
        res = testRadixSort(          
#ifdef RECORDS
        fixedKeys,
#endif
        fixedValues, 
        dataSize, info, tempTime);
    }
    if (tempTime < bestTime && res == true) {
        bestTime = tempTime;
    }
    return bestTime;
}
