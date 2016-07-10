//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "testPrimitives.h"
#include "testJoins.h"
using namespace std;

int dataSize;           //max: MAX_DATA_SIZE
bool is_input;          //whether to read data or fast run
PlatInfo info;          //platform configuration structure

char input_arr_dir[500];
char input_rec_dir[500];
char input_loc_dir[500];

Record *fixedRecords;

int *fixedKeys;
int *fixedValues;

int *fixedLoc;

double runMap(int expeTime, int& blockSize, int& gridSize);
double runGather(int experTime, int& blockSize, int& gridSize);
double runScatter(int experTime, int& blockSize, int& gridSize);
double runScan(int experTime, int& blockSize);
double runRadixSort(int experTime);

/*parameters:
 * if IS_INPUT==true,  executor INPUT_REC_DIR INPUT_ARR_DIR INPUT_LOC_DIR
 * else                executor DATASIZE
 * DATASIZE : data size
 * IS_INPUT : whether to input file from the file system
 * INPUT_REC_DIR : input directory of the record data if needed
 * INPUT_ARR_DIR : input directory of the array data if needed
 * INPUT_LOC_DIR : input directory of the location data if needed
 */
int main(int argc, const char * argv[]) {

    //platform initialization
    PlatInit* myPlatform = PlatInit::getInstance(0);
    cl_command_queue queue = myPlatform->getQueue();
    cl_context context = myPlatform->getContext();
    cl_command_queue currentQueue = queue;
    
    info.context = context;
    info.currentQueue = currentQueue;
    
    switch (argc) {
        case 2:         //fast run
            is_input = false;
            break;
        case 4:         //input
            is_input = true;
            break;
        default:
            cerr<<"Wrong number of parameters."<<endl;
            exit(1);
            break;
    }
    
    if (is_input) {
        strcat(input_rec_dir, argv[1]);
        strcat(input_arr_dir, argv[2]);
        strcat(input_loc_dir, argv[3]);
        std::cout<<"Start reading data..."<<std::endl;
        readFixedRecords(fixedRecords, input_rec_dir, dataSize);
        readFixedArray(fixedLoc, input_loc_dir, dataSize);
        std::cout<<"Finish reading data..."<<std::endl;
    }
    else {
        dataSize = atoi(argv[1]);

    #ifdef RECORDS
        fixedKeys = new int[dataSize];
    #endif
        fixedValues = new int[dataSize];
        fixedLoc = new int[dataSize];
    #ifdef RECORDS
        recordRandom<int>(fixedKeys, fixedValues, dataSize);
    #else
        valRandom<int>(fixedValues,dataSize, MAX_NUM);
    #endif
        valRandom_Only<int>(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
    }
    

    int map_blockSize = -1, map_gridSize = -1;
    int gather_blockSize = -1, gather_gridSize = -1;
    int scatter_blockSize = -1, scatter_gridSize = -1;
    int scan_blockSize = -1;

    int experTime = 10;
    double mapTime = 0.0f, gatherTime = 0.0f, scatterTime = 0.0f, scanTime = 0.0f, radixSortTime = 0.0f;

    // mapTime = runMap(experTime, map_blockSize, map_gridSize);
    // gatherTime = runGather(experTime, gather_blockSize, gather_gridSize);
    // scatterTime = runScatter(experTime, scatter_blockSize, scatter_gridSize);
    scanTime = runScan(experTime, scan_blockSize);
    // radixSortTime = runRadixSort(experTime);

    cout<<"Time for map: "<<mapTime<<" ms."<<'\t'<<"BlockSize: "<<map_blockSize<<'\t'<<"GridSize: "<<map_gridSize<<endl;
    cout<<"Time for gather: "<<gatherTime<<" ms."<<'\t'<<"BlockSize: "<<gather_blockSize<<'\t'<<"GridSize: "<<gather_gridSize<<endl;
    cout<<"Time for scatter: "<<scatterTime<<" ms."<<'\t'<<"BlockSize: "<<scatter_blockSize<<'\t'<<"GridSize: "<<scatter_gridSize<<endl;
    cout<<"Time for scan: "<<scanTime<<" ms."<<'\t'<<"BlockSize: "<<scan_blockSize<<endl;
    cout<<"Time for radix sort: "<<radixSortTime<<" ms."<<endl;
    
//    testSplit(fixedRecords, dataSize, info, 20, totalTime);           //fanout: 20
//    testBitonitSort(fixedRecords, dataSize, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, dataSize, info, 0, totalTime);      //0:  descending
    
//test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    testHj(num, num, info, 16, totalTime);         //16: lower 16 bits to generate the buckets

    return 0;
}

//for testing
#define MIN_BLOCK   (1024)  //64
#define MAX_BLOCK   (1024)
#define MIN_GRID    (1024)  //256
#define MAX_GRID    (1024)

double runMap(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                //--------test map------------
                res = testMap(                
#ifdef RECORDS
                fixedKeys,
#endif
                fixedValues, 
                dataSize,info,tempTime, blockSize, gridSize);
            }
            if (tempTime < bestTime && res == true) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }
    return bestTime;
}

double runGather(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                // --------test gather------------
                res = testGather(            
#ifdef RECORDS
                fixedKeys,
#endif
                fixedValues, 
                dataSize,  info , tempTime, blockSize, gridSize);
            }
            if (tempTime < bestTime && res == true) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }

    return bestTime;
}

double runScatter(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {  
                // --------test scatter------------     
                res = testScatter(             
#ifdef RECORDS
                fixedKeys,
#endif
                fixedValues, 
                dataSize,  info , tempTime, blockSize, gridSize);
            }
            if (tempTime < bestTime && res == true) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }

    return bestTime;
}

double runScan(int experTime, int& bestBlockSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bool res;
    // for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {   
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {   
        double tempTime = MAX_TIME;
        for(int i = 0 ; i < experTime; i++) {       
            // --------test scan------------
            res = testScan(fixedValues, dataSize, info, tempTime, 0, blockSize);             //0: inclusive
        }
        if (tempTime < bestTime && res == true) {
            bestTime = tempTime;
            bestBlockSize = blockSize;
        }
    }
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
