//
//  main.cpp
//  comparison_gpu
//
//  Created by Bryan on 01/20/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "generation.h"

int dataSize = 0;
bool sorted, distinct;

char rec_dir[500];
char arr_dir[500];
char loc_dir[500];

Record *fixedRecords;
int *fixedArray;
int *fixedLoc;

void generateData(int dataSize, bool sorted, bool distinct) {
	if (sorted) {
		if (distinct) {
			recordSorted_Only(fixedRecords, dataSize, SHUFFLE_TIME(dataSize));
			intSorted_Only(fixedArray, dataSize, SHUFFLE_TIME(dataSize));
		}
		else {
			recordSorted(fixedRecords,dataSize);
			intSorted(fixedArray,dataSize);
		}
	}
	else {
		if (distinct) {
			recordRandom_Only(fixedRecords,dataSize, SHUFFLE_TIME(dataSize));
			intRandom_Only(fixedArray,dataSize, SHUFFLE_TIME(dataSize));
		}
		else {
			recordRandom(fixedRecords, dataSize);
			intRandom(fixedArray, dataSize);
		}
	}
	intRandom_Only(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
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
	dataSize = atoi(argv[4]);
	if (dataSize > MAX_DATA_SIZE) {
		std::cerr<<"Data size too large."<<std::endl;
		exit(1);
	} 

	fixedRecords = new Record[dataSize];
	fixedArray = new int[dataSize];
	fixedLoc = new int[dataSize];

	sorted = processBool(argv[5]);
	distinct = processBool(argv[6]);

	if (strcmp(argv[1],"none"))	strcat(rec_dir,argv[1]);
	if (strcmp(argv[2],"none"))	strcat(arr_dir,argv[2]);
	if (strcmp(argv[3],"none"))	strcat(loc_dir,argv[3]);

	generateData(dataSize, sorted, distinct);

	writeRecords(fixedRecords, dataSize, rec_dir);
	writeArray(fixedArray, dataSize, arr_dir);
	writeArray(fixedLoc, dataSize, loc_dir);

	delete[] fixedRecords;
	delete[] fixedArray;
	delete[] fixedLoc;
	
	return 0;
}