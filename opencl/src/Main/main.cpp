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
double runScan(int experTime, int& blockSize);
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
    dataSize = atoi(argv[1]);
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

    int experTime = 10;
    double mapTime = 0.0f, gatherTime = 0.0f, scatterTime = 0.0f, scanTime = 0.0f, radixSortTime = 0.0f;

    double totalTime;
//    dataSize = 160000000;

//    testMem(info);
//     testAccess(info);

    // runMem<double>();
    // runAtomic();
    // runBarrier(experTime);
    // testLatency(info);

     // runMap();
     // testGather(fixedValues, 1000000000, info);
    // testScatter(fixedValues, 1000000000, info);
    // radixSortTime = runRadixSort(experTime);

//    scanTime = runScan(experTime, scan_blockSize);
//    cout<<"Time for scan: "<<scanTime<<" ms."<<'\t'<<"BlockSize: "<<scan_blockSize<<endl;
    // cout<<"Time for radix sort: "<<radixSortTime<<" ms."<<endl;

//    testSplit(dataSize, info, 12, totalTime);

//    testBitonitSort(fixedRecords, dataSize, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, dataSize, info, 0, totalTime);      //0:  descending
    
//test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    int num = 1600000;
    testHj(dataSize, dataSize, info);         //16: lower 16 bits to generate the buckets

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

void runAtomic() {

	std::cout<<"----- Local Atomic Test -----"<<std::endl;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    // int blockIdx = 0, gridIdx = 0;
    // for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
    // 	gridIdx = 0;
    //     for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
    //         	double localTime;

    //             //--------test local atomic------------
    //             testAtomic(                
    //           	info,localTime, blockSize, gridSize, true);

    //         gridIdx++;
    //     }
    //     blockIdx++;
    // }

    for(int gz = 1; gz <= 1024; gz++) {
        for(int r = 1; r <=3; r++) {
            double localTime;
            testAtomic(info,localTime, 512, gz, false);
        }
        
    }

    // std::cout<<"----- Global Atomic Test -----"<<std::endl;

    // blockIdx = 0, gridIdx = 0;
    // for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
    //     gridIdx = 0;
    //     for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
    //         	double localTime;

    //             //--------test global atomic------------
    //             testAtomic(                
    //           	info,localTime, blockSize, gridSize, false);

    //         gridIdx++;
    //     }
    //     blockIdx++;
    // }
}

double runMap() {
    int blockSize = 1024, gridSize = 8192;
    int repeat = 64, repeat_trans = 16;
    testMap(info, repeat, repeat_trans, blockSize, gridSize);
}

double runScan(int experTime, int& bestBlockSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bool res;

    int block_min = 1024, block_max = 1024;
    int grid_min = 1024, grid_max = 1024;
    
    //we only test exclusive
    cout<<"--------- Exclusive --------"<<endl;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {   
        double tempTime = MAX_TIME;
        for(int i = 0 ; i < experTime; i++) {       
            // --------test scan------------
            res = testScan(fixedValues, dataSize, info, tempTime, 1, blockSize);             //0: inclusive
        }
        if (tempTime < bestTime && res == true) {
            bestTime = tempTime;
            bestBlockSize = blockSize;
        }
    }

    // cout<<"--------- Inclusive --------"<<endl;
    // for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {   
    //     double tempTime = MAX_TIME;
    //     for(int i = 0 ; i < experTime; i++) {       
    //         // --------test scan------------
    //         res = testScan(fixedValues, dataSize, info, tempTime, 0, blockSize);             //0: inclusive
    //     }
    //     if (tempTime < bestTime && res == true) {
    //         bestTime = tempTime;
    //         bestBlockSize = blockSize;
    //     }
    // }
    return bestTime;
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
