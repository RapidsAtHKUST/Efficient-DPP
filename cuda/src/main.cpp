//
//  main.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "test.h"

using namespace std;

char input_rec_dir[500];
char input_arr_dir[500];
char input_loc_dir[500];

bool is_input;
int dataSize;
int fanout = 10;

Record *fixedRecords;
int *fixedArray;
int *fixedLoc;

int *fixedKeys;
int *fixedValues;

int *splitVals;
int *splitKeys;
int *splitArray;	

int recordLength;
int arrayLength;

int main(int argc, char *argv[]) {
 
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

#ifdef RECORDS
    std::cout<<"Using type: Record"<<std::endl;
#else
    std::cout<<"Using type: Basic type"<<std::endl;
#endif

    if (is_input) {
        strcat(input_rec_dir, argv[1]);
        strcat(input_arr_dir, argv[2]);
        strcat(input_loc_dir, argv[3]);
        std::cout<<"Start reading data..."<<std::endl;
        readFixedRecords(fixedRecords, input_rec_dir, dataSize);
        readFixedArray(fixedArray, input_arr_dir, dataSize);
        readFixedArray(fixedLoc, input_loc_dir, dataSize);
        std::cout<<"Finish reading data..."<<std::endl;
    }
    else {
        dataSize = atoi(argv[1]);
        
        fixedRecords = new Record[dataSize];
        fixedArray = new int[dataSize];
        fixedLoc = new int[dataSize];

        fixedKeys = new int[dataSize];
        fixedValues = new int[dataSize];

        splitVals = new int[dataSize];
		splitKeys = new int[dataSize];
		splitArray = new int[dataSize];

        recordRandom<int>(fixedKeys, fixedValues, dataSize, MAX_NUM);
		recordRandom<int>(splitKeys, splitVals, dataSize, fanout);

        valRandom<int>(fixedArray, dataSize, 10);
        // valRandom<int>(fixedArray, dataSize, MAX_NUM);

        valRandom_Only<int>(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
		valRandom<int>(splitArray, dataSize, fanout);			//fanout
    }

	bool res;
	int experiNum = 10;

	int blockSize = 1024, gridSize = 32768;
//-------------------------------- Basic test-----------------------------
	// float bestTime_read, bestTime_write, bestTime_mul;
	// float throughput_read, throughput_write, throughput_mul;
	// //basic test
	// int repeat_read = 100;
	// testMem(blockSize, gridSize, bestTime_read, bestTime_write, bestTime_mul, repeat_read);

	// throughput_read = computeMem(blockSize*gridSize/4, sizeof(int), bestTime_read);
 //    throughput_write = computeMem(blockSize*gridSize, sizeof(int), bestTime_write);
 //    throughput_mul = computeMem(blockSize*gridSize*2*2, sizeof(int), bestTime_mul);

 //    cout<<"Time for memory read(Repeat:"<<repeat_read<<"): "<<bestTime_read<<" ms."<<'\t'
 //        <<"Bandwidth: "<<throughput_read<<" GB/s"<<endl;

 //    cout<<"Time for memory write: "<<bestTime_write<<" ms."<<'\t'
 //        <<"Bandwidth: "<<throughput_write<<" GB/s"<<endl;

 //    cout<<"Time for memory mul: "<<bestTime_mul<<" ms."<<'\t'
 //    <<"Bandwidth: "<<throughput_mul<<" GB/s"<<endl;
//-------------------------------- Basic test end-----------------------------


	//total time for each primitive
	float idnElapsedTime;
	float mapTime = MAX_TIME;
	float gatherTime = MAX_TIME;
	float scatterTime = MAX_TIME;
	float splitTime = MAX_TIME;
	float scanTime = MAX_TIME;
	float scanTime_ble = MAX_TIME;
	float radixSortTime = MAX_TIME;

	for(int i = 0; i < experiNum; i++) {
		// cout<<"Round "<<i<<" :"<<endl;

// //--------------testing map--------------
// 		res = testMap<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, idnElapsedTime);

// 		printRes("map", res,idnElapsedTime);
// 		if (idnElapsedTime < mapTime)		mapTime = idnElapsedTime;

// //--------------testing gather--------------
		res = testGather<int>(
#ifdef RECORDS
		fixedKeys, fixedValues,
#else
		fixedArray, 
#endif
		dataSize, fixedLoc, idnElapsedTime, blockSize, gridSize);

		printRes("gather", res,idnElapsedTime);
		if (idnElapsedTime < gatherTime)		gatherTime = idnElapsedTime;
		

// //--------------testing scatter--------------
// 		res = testScatter<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, idnElapsedTime);

// 		printRes("scatter", res,idnElapsedTime);
// 		if (idnElapsedTime < scatterTime)		scatterTime = idnElapsedTime;

// // --------------testing split--------------
// 		//use blockSize of 256 in fix
// 		res = testSplit<int>(
// #ifdef RECORDS
// 		splitKeys, splitVals,
// #else
// 		splitArray, 
// #endif
// 		dataSize, idnElapsedTime, fanout);

// 		printRes("split", res,idnElapsedTime);
// 		if (idnElapsedTime < splitTime)		splitTime = idnElapsedTime;

//--------------testing scan: 0 for inclusive, 1 for exclusive--------------

		// res = testScan_warp<int>(fixedArray, dataSize, idnElapsedTime, 1);
		// printRes("scan_warp", res,idnElapsedTime);
		// if (idnElapsedTime < scanTime)		scanTime = idnElapsedTime;

		// res = testScan_ble<int>(fixedArray, dataSize, idnElapsedTime, 1);
		// printRes("scan_ble", res,idnElapsedTime);
		// if (idnElapsedTime < scanTime_ble)		scanTime_ble = idnElapsedTime;
		// scanImpl(fixedArray, dataSize, BLOCKSIZE, GRIDSIZE, 1);
//--------------testing radix sort (no need to specify the block and grid size)--------------

// 		res = testRadixSort<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, idnElapsedTime);

// 		printRes("radix sort", res,idnElapsedTime);
// 		if (idnElapsedTime < radixSortTime)		radixSortTime = idnElapsedTime;
	}
	gatherTime = gatherTime / dataSize * 1e6;		//get per tuple result

	cout<<"-----------------------------------------"<<endl;
#ifdef RECORDS
    cout<<"Using type: Record."<<endl;
#else
    cout<<"Using type: Basic type."<<endl;
#endif
	cout<<"Data Size: "<<dataSize<<endl;
	cout<<"Map time: "<<mapTime<<" ns."<<endl;
	cout<<"Gather time per tuple: "<<gatherTime<<" ns."<<endl;
	cout<<"Scatter time: "<<scatterTime<<" ns."<<endl;
	cout<<"Split time: "<<splitTime<<" ms."<<endl;
	cout<<"Scan time:"<<scanTime<<" ms."<<endl;
	cout<<"Scan ble time:"<<scanTime_ble<<" ms."<<endl;
	cout<<"Radix sort time:"<<radixSortTime<<" ms."<<endl;
	
	delete[] fixedRecords;
	delete[] fixedArray;
	delete[] fixedLoc;

	delete[] fixedKeys;
	delete[] fixedValues;

	delete[] splitVals;
	delete[] splitKeys;
	delete[] splitArray;

	return 0;
}
