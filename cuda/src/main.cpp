//
//  main.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "test.h"

using namespace std;

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

	dataSize = atoi(argv[1]);

	//basic test
    cudaSetDevice(0);
	testMem();

	// throughput_read = computeMem(blockSize*gridSize/4, sizeof(int), bestTime_read);
 //    throughput_write = computeMem(blockSize*gridSize, sizeof(int), bestTime_write);
 //    throughput_mul = computeMem(blockSize*gridSize*2*2, sizeof(int), bestTime_mul);
 //    throughput_add = computeMem(blockSize*gridSize*2*3, sizeof(int), bestTime_add);

 //    cout<<"Time for memory read(Repeat:"<<repeat_read<<"): "<<bestTime_read<<" ms."<<'\t'
 //        <<"Bandwidth: "<<throughput_read<<" GB/s"<<endl;

 //    cout<<"Time for memory write: "<<bestTime_write<<" ms."<<'\t'
 //        <<"Bandwidth: "<<throughput_write<<" GB/s"<<endl;

 //    cout<<"Time for memory mul: "<<bestTime_mul<<" ms."<<'\t'
 //    <<"Bandwidth: "<<throughput_mul<<" GB/s"<<endl;
	
	// cout<<"Time for memory add: "<<bestTime_add<<" ms."<<'\t'
 //    <<"Bandwidth: "<<throughput_add<<" GB/s"<<endl;
//-------------------------------- Basic test end-----------------------------


	//total time for each primitive

//	for(int i = 0; i < experiNum; i++) {
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
// 		res = testGather<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, idnElapsedTime, blockSize, gridSize);

// 		printRes("gather", res,idnElapsedTime);
// 		if (idnElapsedTime < gatherTime)		gatherTime = idnElapsedTime;
		

// //--------------testing scatter--------------
//		res = testScatter<int>(
//#ifdef RECORDS
//		fixedKeys, fixedValues,
//#else
//		fixedArray,
//#endif
//		dataSize, fixedLoc, idnElapsedTime);
//
//		printRes("scatter", res,idnElapsedTime);
//		if (idnElapsedTime < scatterTime)		scatterTime = idnElapsedTime;

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
//	}


	return 0;
}
