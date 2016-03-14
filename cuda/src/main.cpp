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

Record *fixedRecords;
int *fixedArray;
int *fixedLoc;

int *fixedKeys;
int *fixedValues;

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

        recordRandom<int>(fixedKeys, fixedValues, dataSize, MAX_NUM);
        valRandom<int>(fixedArray, dataSize, MAX_NUM);
        valRandom_Only<int>(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
    }

	float totalTime = 0.0f;
	bool res;


// 	res = testMap<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, totalTime);

	// cout<<"map: ";
	// if (res) 	cout<<"Success!"<<'\t';
	// else 		cout<<"Fail!"<<'\t';
	// cout<<"Time: "<<totalTime<<" ms"<<endl;

	int experiNum = 100;

	//total time for each primitive
	double gatherTotal = 0.0f;
	double gather_mul_total = 0.0f;

	double mapTotal = 0.0f;
	double scatterTotal = 0.0f;
	double splitTotal = 0.0f;
	double scanTotal = 0.0f;
	double radixSortTotal = 0.0f;


	for(int i = 0; i < experiNum; i++) {
		// res = testMap(fixedRecords, dataSize, totalTime);
		// cout<<"map["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// mapTotal += totalTime;

// 	res = testGather<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, totalTime);

// 	cout<<"gather: ";
// 	if (res) 	cout<<"Success!"<<'\t';
// 	else 		cout<<"Fail!"<<'\t';
// 	cout<<"Time: "<<totalTime<<" ms"<<endl;

// 		res = testGather_mul<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, totalTime);

// 	cout<<"gather_mul: ";
// 	if (res) 	cout<<"Success!"<<'\t';
// 	else 		cout<<"Fail!"<<'\t';
// 	cout<<"Time: "<<totalTime<<" ms"<<endl;


// 		res = testScatter<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, totalTime);

// 	cout<<"scatter: ";
// 	if (res) 	cout<<"Success!"<<'\t';
// 	else 		cout<<"Fail!"<<'\t';
// 	cout<<"Time: "<<totalTime<<" ms"<<endl;



// 		res = testScatter_mul<int>(
// #ifdef RECORDS
// 		fixedKeys, fixedValues,
// #else
// 		fixedArray, 
// #endif
// 		dataSize, fixedLoc, totalTime);

// 	cout<<"scatter_mul: ";
// 	if (res) 	cout<<"Success!"<<'\t';
// 	else 		cout<<"Fail!"<<'\t';
// 	cout<<"Time: "<<totalTime<<" ms"<<endl;



		int fanout = 20;
		int *splitVals = new int[dataSize];
#ifdef RECORDS
		int *splitKeys = new int[dataSize];
		recordRandom<int>(splitKeys, splitVals, dataSize, fanout);
#else 
		valRandom<int>(splitVals, dataSize, fanout);
#endif
		res = testSplit<int>(
#ifdef RECORDS
		splitKeys, splitVals,
#else
		splitVals, 
#endif
		dataSize, totalTime, fanout);

	cout<<"split: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';

	cout<<"Time: "<<totalTime<<" ms"<<endl;

		// for(int i = 0; i < dataSize; i++)	{
		// 	cout<<splitVals[i]<<' ';
		// }
		// cout<<endl;

		// res = testGather(fixedRecords, dataSize, fixedLoc, totalTime);
		// cout<<"gather["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// gatherTotal += totalTime;

		// res = testGather_mul(fixedRecords, dataSize, fixedLoc, totalTime);
		// cout<<"gather_mul["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// gather_mul_total += totalTime;		
		
		// res = testScatter(fixedRecords, dataSize, fixedLoc,totalTime);
		// cout<<"scatter["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// scatterTotal += totalTime;
		
		// int fanout = 20;
		// Record *records = new Record[dataSize];
		// recordRandom(records, dataSize, fanout);

		// cout<<"split["<<i<<"] finished"<<endl;
		// res = testSplit(records, dataSize, totalTime, fanout);		//fanout = 20
		// if (!res) 	exit(1);
		// if (i != 0)
		// 	splitTotal += totalTime;

		// cout<<"Input:"<<endl;
		// for(int i = 0; i < dataSize; i++) {
		// 	cout<<fixedRecords[i].x<<' '<<fixedRecords[i].y<<endl;
		// }
		
		 // res = testScan(fixedArray, dataSize, totalTime, 1);
		
		// cout<<"scan["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// if (i != 0)		
			// scanTotal += totalTime;

		// res = testBisort(fixedRecords, dataSize, totalTime, 1);
		// if (res)		cout<<"success!"<<endl;
		// else			cout<<"Failed!"<<endl;
		// cout<<"bisort time: "<<totalTime<<" ms."<<endl;

		// cout<<"radixSort: ";
		// int gridSize = 512;
		// int blockSize =1024;
		// res = testRadixSort_int(fixedArray, dataSize, totalTime,blockSize,gridSize);
		// if (res) 	cout<<"Success!"<<'\t';
		// else 		cout<<"Fail!"<<'\t';
		// if (i != 0)
		// 	radixSortTotal += totalTime;
		// cout<<"Time: "<<totalTime<<" ms"<<endl;
	}

	// testRadixSort();
	 // scan_warp_test();

	// cout<<"map avg time: "<<mapTotal/experiNum<<" ms."<<endl;
	// cout<<"gather avg time: "<<gatherTotal/experiNum<<" ms."<<endl;
	// cout<<"gather_mul avg time: "<<gather_mul_total/experiNum<<" ms."<<endl;

	// cout<<"scatter avg time: "<<scatterTotal/experiNum<<" ms."<<endl;

	// cout<<"split avg time: "<<splitTotal/(experiNum-1)<<" ms."<<endl;
	// testScan(fixedArray, dataSize, totalTime,1);

	// cout<<"My Scan Time:"<<scanTotal/experiNum<<" ms."<<endl;
	// cout<<"Radix sort Time:"<<radixSortTotal/(experiNum-1)<<" ms."<<endl;

	
	
	// if (res) 	cout<<"Success!"<<'\t';
	// else 		cout<<"Fail!"<<'\t';
	// cout<<"Time: "<<totalTime<<" ms"<<endl;
	// delete[] records;

	
	
	delete[] fixedRecords;
	delete[] fixedArray;
	delete[] fixedLoc;

	return 0;

}
