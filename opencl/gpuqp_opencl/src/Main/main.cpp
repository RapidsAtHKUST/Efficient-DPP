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

bool is_random_generated = true;
bool is_output = false;
bool is_input = false;
int dataSize;

char output_rec_dir[300];
char output_arr_dir[300];
char input_arr_dir[300];
char input_rec_dir[300];

Record fixedRecords[MAX_DATA_SIZE];
int fixedArray[MAX_DATA_SIZE];

int recordLength;           //max: MAX_DATA_SIZE
int arrayLength;            //max: MAX_DATA_SIZE

/*parameters:
 * if IS_INPUT==true,  gpuqp DATASIZE IS_RANDOM_GENERATED IS_OUTPUT IS_INPUT INPUT_REC_DIR INPUT_ARR_DIR
 * if IS_OUTPUT==true, gpuqp DATASIZE IS_RANDOM_GENERATED IS_OUTPUT IS_INPUT OUPUT_REC_DIR OUTPUT_ARR_DIR
 * DATASIZE : data size
 * IS_RANDOM_GENERATED : whether to generate data randomly
 * IS_OUTPUT : whether to output file to the file system
 * OUTPUT_REC_DIR : output directory of record data if needed
 * OUTPUT_ARR_DIR : output directory of array data if needed
 * IS_INPUT : whether to input file from the file system
 * INPUT_REC_DIR : input directory of the record data if needed
 * INPUT_ARR_DIR : input directory of the array data if needed
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
    
    dataSize = atoi(argv[1]);
    recordLength = dataSize;
    arrayLength = dataSize;
    
    //processing $IS_RANDOM_GENERATED
    processBool(argv[2],is_random_generated);
    processBool(argv[3],is_output);
    processBool(argv[4],is_input);
    
    if (is_output) {
        strcat(output_rec_dir, argv[5]);
        strcat(output_arr_dir, argv[6]);
    }
    if (is_input) {
        strcat(input_rec_dir, argv[5]);
        strcat(input_arr_dir, argv[6]);
        readFixedRecords(fixedRecords, input_rec_dir, recordLength);
        readFixedArray(fixedArray, input_arr_dir, arrayLength);
    }
    
    if (is_random_generated) {
        generateFixedRecords(fixedRecords, dataSize, is_output, output_rec_dir);
        generateFixedArray(fixedArray, dataSize, is_output, output_arr_dir);
    }
    
    //test primitives
    testMap(fixedRecords, recordLength, info, totalTime);
//    testGather(fixedRecords, recordLength, info, totalTime);
//    testScatter(fixedRecords, recordLength, info, totalTime);
//    testScan(fixedArray, arrayLength, info, totalTime, 0);             //0: inclusive
//    testScan(fixedArray, arrayLength, info, totalTime, 1);             //1: exclusive
//    testSplit(fixedRecords, recordLength, info, 20, totalTime);           //fanout: 20
//    testRadixSort(fixedRecords, recordLength, info, totalTime);
//    testBitonitSort(fixedRecords, recordLength, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, recordLength, info, 0, totalTime);      //0:  descending
    
    //test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    testHj(num, num, info, 16, totalTime);         //16: lower 16 bits to generate the buckets

    return 0;
}
