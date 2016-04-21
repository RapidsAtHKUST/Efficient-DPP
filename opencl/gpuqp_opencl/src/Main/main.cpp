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

char input_arr_dir[500];
char input_rec_dir[500];
char input_loc_dir[500];

Record *fixedRecords;

int *fixedKeys;
int *fixedValues;

int *fixedLoc;

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
    
    PlatInfo info;
    info.context = context;
    info.currentQueue = currentQueue;
    
    double totalTime = 0;
    
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
    
    //test primitives
    for(int i= 0 ; i < 10; i++) {

//     testMap(
// #ifdef RECORDS
//     fixedKeys,
// #endif
//     fixedValues, 
//     dataSize,  info , totalTime);

//     testGather(
// #ifdef RECORDS
//     fixedKeys,
// #endif
//     fixedValues, 
//     dataSize,  info , totalTime);
//     testScatter(
// #ifdef RECORDS
//     fixedKeys,
// #endif
//     fixedValues, 
//     dataSize,  info , totalTime);

//     cout<<"finished "<<i<<endl;
   testScan(fixedValues, dataSize, info, totalTime, 0);             //0: inclusive
        
    }
    
   // testMap(fixedRecords, dataSize, info, totalTime);
	// testGather(fixedRecords, dataSize, info, totalTime);
	// testScatter(fixedRecords, dataSize, info, totalTime);
//    testScan(fixedArray, dataSize, info, totalTime, 1);             //1: exclusive
//    testSplit(fixedRecords, dataSize, info, 20, totalTime);           //fanout: 20
    // testRadixSort(fixedRecords, dataSize, info, totalTime);
    
    // testBitonitSort(fixedRecords, dataSize, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, dataSize, info, 0, totalTime);      //0:  descending
    
    //test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    testHj(num, num, info, 16, totalTime);         //16: lower 16 bits to generate the buckets

    return 0;
}
