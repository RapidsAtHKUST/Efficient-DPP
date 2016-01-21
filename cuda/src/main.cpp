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
        readFixedRecords(fixedRecords, input_rec_dir, dataSize);
        readFixedArray(fixedArray, input_arr_dir, dataSize);
        readFixedArray(fixedLoc, input_loc_dir, dataSize);
    }
    else {
        dataSize = atoi(argv[1]);
        recordRandom(fixedRecords, dataSize);
        intRandom(fixedArray, dataSize, MAX_NUM);
        intRandom_Only(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
    }

	Record *source = fixedRecords;
	// recordRandom(source, r_len);

	int *loc = fixedLoc;
	// intRandom_Only(loc, r_len,10000);

	double time = 0.0f;

// cout<<"Records:"<<endl;
// for(int i = 0; i < recordLength; i++) {
// 	cout<<fixedRecords[i].x <<' '<<fixedRecords[i].y<<endl;
// }

// cout<<"Locs:"<<endl;
// for(int i = 0; i < recordLength; i++) {
// 	cout<<fixedLoc[i]<<endl;
// }
// exit(0);

	int r_len = arrayLength;

	bool res = testMap(source, r_len, time);
	cout<<"map: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	res = testGather(source, r_len, loc, time);
	cout<<"gather: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	res = testScatter(source, r_len, loc,time);
	cout<<"scatter: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	// delete[] source;
	// delete[] loc;

	return 0;

}