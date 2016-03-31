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

        valRandom<int>(fixedArray, dataSize, MAX_NUM);
        valRandom_Only<int>(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
		valRandom<int>(splitArray, dataSize, fanout);			//fanout

    }

	bool res;
	bool isFirstCount = false;			//not counting the time in the first loop
	int experiNum = 10;
	int act_experiNum;

	if (isFirstCount)	act_experiNum = experiNum;
	else 				act_experiNum = experiNum - 1;	
	assert(act_experiNum != 0);

	//total time for each primitive
	float idnElapsedTime;
	float mapTotal = 0.0f;
	float gatherTotal = 0.0f;
	float scatterTotal = 0.0f;
	float splitTotal = 0.0f;
	float scanTotal = 0.0f;
	float radixSortTotal = 0.0f;

	for(int i = 0; i < experiNum; i++) {
		cout<<"Round "<<i<<" :"<<endl;

//--------------testing map--------------
		res = testMap<int>(
#ifdef RECORDS
		fixedKeys, fixedValues,
#else
		fixedArray, 
#endif
		dataSize, idnElapsedTime);

		printRes("map", res,idnElapsedTime);
		if (i != 0)		mapTotal += idnElapsedTime;

//--------------testing gather--------------
		res = testGather<int>(
#ifdef RECORDS
		fixedKeys, fixedValues,
#else
		fixedArray, 
#endif
		dataSize, fixedLoc, idnElapsedTime);

		printRes("gather", res,idnElapsedTime);
		if (i != 0)		gatherTotal += idnElapsedTime;

//--------------testing scatter--------------
		res = testScatter<int>(
#ifdef RECORDS
		fixedKeys, fixedValues,
#else
		fixedArray, 
#endif
		dataSize, fixedLoc, idnElapsedTime);

		printRes("scatter", res,idnElapsedTime);
		if (i != 0)		scatterTotal += idnElapsedTime;

//--------------testing split--------------
		res = testSplit<int>(
#ifdef RECORDS
		splitKeys, splitVals,
#else
		splitArray, 
#endif
		dataSize, idnElapsedTime, fanout);

		printRes("split", res,idnElapsedTime);
		if (i != 0)		splitTotal += idnElapsedTime;

//--------------testing scan: 0 for inclusive, 1 for exclusive--------------
		res = testScan<int>(fixedArray, dataSize, idnElapsedTime, 0);
		printRes("scan", res,idnElapsedTime);
		if (i != 0)		scanTotal += idnElapsedTime;

//--------------testing radix sort (no need to specify the block and grid size)--------------
		res = testRadixSort<int>(
#ifdef RECORDS
		fixedKeys, fixedValues,
#else
		fixedArray, 
#endif
		dataSize, idnElapsedTime);

		printRes("radix sort", res,idnElapsedTime);
		if (i != 0)		radixSortTotal += idnElapsedTime;
	}

	cout<<"-----------------------------------------"<<endl;
#ifdef RECORDS
    cout<<"Using type: Record."<<endl;
#else
    cout<<"Using type: Basic type."<<endl;
#endif
	cout<<"Data Size: "<<dataSize<<endl;
	cout<<"Map avg time: "<<mapTotal/act_experiNum<<" ms."<<endl;
	cout<<"Gather avg time: "<<gatherTotal/act_experiNum<<" ms."<<endl;
	cout<<"Scatter avg time: "<<scatterTotal/act_experiNum<<" ms."<<endl;
	cout<<"Split avg time: "<<splitTotal/act_experiNum<<" ms."<<endl;
	cout<<"Scan avg time:"<<scanTotal/act_experiNum<<" ms."<<endl;
	cout<<"Radix sort avg time:"<<radixSortTotal/act_experiNum<<" ms."<<endl;
	
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
