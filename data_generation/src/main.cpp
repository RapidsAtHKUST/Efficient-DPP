//
//  main.cpp
//  comparison_gpu
//
//  Created by Bryan on 01/20/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "generation.h"
#include <iostream>

#define SHUFFLE_NUM		(2099999999)

int dataSize = 0;
bool sorted, distinct;

char rec_dir[500];
char arr_dir[500];
char loc_dir[500];

Record *fixedRecords;
int *fixedArray;
int *fixedLoc;

void my_itoa(int num, char *buffer, int base) {
    int len=0, p=num;
    while(p/=base)
    {
        len++;
    }
    len++;
    for(p=0;p<len;p++)
    {
        int x=1;
        for(int t=p+1;t<len;t++)
        {
            x*=base;
        }
        buffer[p] = num/x + '0';
        num -=( buffer[p] - '0' ) * x;
    }
    buffer[len] = '\0';
}

void generateData(int dataSize, bool sorted, bool distinct) {
	// if (sorted) {
	// 	if (distinct) {
	// 		recordSorted_Only(fixedRecords, dataSize, SHUFFLE_TIME(dataSize));
	// 		intSorted_Only(fixedArray, dataSize, SHUFFLE_TIME(dataSize));
	// 	}
	// 	else {
	// 		recordSorted(fixedRecords,dataSize);
	// 		intSorted(fixedArray,dataSize);
	// 	}
	// }
	// else {
	// 	if (distinct) {
	// 		recordRandom_Only(fixedRecords,dataSize, SHUFFLE_TIME(dataSize));
	// 		intRandom_Only(fixedArray,dataSize, SHUFFLE_TIME(dataSize));
	// 	}
	// 	else {
	// 		recordRandom(fixedRecords, dataSize);
	// 		intRandom(fixedArray, dataSize);
	// 	}
	// }
	intRandom_Only(fixedLoc, dataSize, SHUFFLE_NUM, dataSize);
}

/*
 *	data_executor $REC_DIR $ARR_DIR $LOC_DIR $DATASIZE $SORTED $DISTINCT 
 *
 */
int main(int argc, char* argv[]) {

	if (argc != 7) {
		std::cerr<<"Wrong number of Parameters."<<std::endl;
		exit(1);
	}

	//parsing the parameters
	// dataSize = atoi(argv[4]);
	// if (dataSize > MAX_DATA_SIZE) {
	// 	std::cerr<<"Data size too large."<<std::endl;
	// 	exit(1);
	// } 
	dataSize = 1000000000;

	// fixedRecords = new Record[dataSize];
	// fixedArray = new int[dataSize];
	fixedLoc = new int[dataSize];

	sorted = processBool(argv[5]);
	distinct = processBool(argv[6]);

	if (strcmp(argv[1],"none"))	strcat(rec_dir,argv[1]);
	if (strcmp(argv[2],"none"))	strcat(arr_dir,argv[2]);
	if (strcmp(argv[3],"none"))	strcat(loc_dir,argv[3]);

	int myDataSize[17] = {1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 100000000,200000000,300000000,400000000,500000000,600000000,700000000,800000000,900000000,1000000000};

	for(int i = 14; i < 15; i++) {
		std::cout<<"now making: "<<myDataSize[i]<<std::endl;
		generateData(myDataSize[i], sorted, distinct);
		std::cout<<"generating finished. Start writing... "<<std::endl;

		char tempNum[40];
		my_itoa(myDataSize[i], tempNum, 10);

		char loc_dir[500]={'\0'};
		strcat(loc_dir,"loc_");
		strcat(loc_dir, tempNum);
		strcat(loc_dir, "_equal.data");

		writeArray(fixedLoc, myDataSize[i], loc_dir);
		std::cout<<"generate "<<myDataSize[i]<<" finished."<<std::endl;
	}

	// writeRecords(fixedRecords, dataSize, rec_dir);
	// writeArray(fixedArray, dataSize, arr_dir);
	// writeArray(fixedLoc, 1000000, loc_dir);

	// delete[] fixedRecords;
	// delete[] fixedArray;
	delete[] fixedLoc;
	
	return 0;
}
