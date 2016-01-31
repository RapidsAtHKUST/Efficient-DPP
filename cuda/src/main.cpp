//
//  main.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "test.h"
#include <unistd.h>

using namespace std;

char input_rec_dir[500];
char input_arr_dir[500];
char input_loc_dir[500];

bool is_input;
int dataSize;

Record fixedRecords[MAX_DATA_SIZE];
int fixedArray[MAX_DATA_SIZE];
int fixedLoc[MAX_DATA_SIZE];

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
        recordRandom(fixedRecords, dataSize);
        intRandom(fixedArray, dataSize, MAX_NUM);
        intRandom_Only(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
    }

	double totalTime = 0.0f;
	bool res;

	// res = testMap(fixedRecords, dataSize, totalTime);
	// cout<<"map: ";
	// if (res) 	cout<<"Success!"<<'\t';
	// else 		cout<<"Fail!"<<'\t';
	// cout<<"Time: "<<totalTime<<" ms"<<endl;

	int experiNum = 1;

	//total time for each primitive
	double gatherTotal = 0.0f;
	double scatterTotal = 0.0f;
	double splitTotal = 0.0f;

	for(int i = 0; i < experiNum; i++) {
		// res = testGather(fixedRecords, dataSize, fixedLoc, totalTime);
		// cout<<"gather["<<i<<"] finished"<<endl;
		// if (!res) 	exit(1);
		// gatherTotal += totalTime;

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
		// splitTotal += totalTime;

		// cout<<"Input:"<<endl;
		// for(int i = 0; i < dataSize; i++) {
		// 	cout<<fixedRecords[i].x<<' '<<fixedRecords[i].y<<endl;
		// }
		
		res = testBisort(fixedRecords, dataSize, totalTime, 1);
		if (res)		cout<<"success!"<<endl;
		else			cout<<"Failed!"<<endl;
		cout<<"bisort time: "<<totalTime<<" ms."<<endl;
	}
	// cout<<"gather avg time: "<<gatherTotal/experiNum<<" ms."<<endl;
	// cout<<"scatter avg time: "<<scatterTotal/experiNum<<" ms."<<endl;

	// cout<<"split avg time: "<<splitTotal/experiNum<<" ms."<<endl;
	// testScan(fixedArray, dataSize, totalTime,1);
	// cout<<"My Scan Time:"<<totalTime<<" ms."<<endl;

	
	
	// if (res) 	cout<<"Success!"<<'\t';
	// else 		cout<<"Fail!"<<'\t';
	// cout<<"Time: "<<totalTime<<" ms"<<endl;
	// delete[] records;

	// cout<<"radixSort: ";
	// res = testRadixSort(fixedRecords, dataSize, totalTime);
	// if (res) 	cout<<"Success!"<<'\t';
	// else 		cout<<"Fail!"<<'\t';
	// cout<<"Time: "<<totalTime<<" ms"<<endl;
	
	return 0;

}