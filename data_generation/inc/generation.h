//
//  generation.h
//  comparison_gpu
//
//  Created by Bryan on 01/20/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//

#ifndef __GENERATION_H__
#define __GENERATION_H__

#include <iostream>
#include <fstream>

#include <climits>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <string.h>

#include <cuda_runtime.h>

#define SHUFFLE_TIME(TIME)  		(TIME*2)
#define MAX_DATA_SIZE				(160000000)
#define MAX_NUM 					(INT_MAX/2)

typedef int2 Record;

void recordRandom(Record *records, int length, int max = MAX_NUM);
void recordRandom_Only(Record *records, int length,  int times);
void recordSorted(Record *records, int length, int max = MAX_NUM);
void recordSorted_Only(Record *records, int length, int times);

void intRandom(int *intArr, int length, int max = MAX_NUM);
void intRandom_Only(int *intArr, int length,  int times);

void intSorted(int *intArr, int length, int max = MAX_NUM);
void intSorted_Only(int *intArr, int length, int times);

//generate the fixed record & int array and write to the external memory.
void writeRecords(Record* records, int length, char *file);
void writeArray(int *intArr, int length, char *file);


int compRecordAsc ( const void * a, const void * b);
int compRecordDec ( const void * a, const void * b);
int compInt ( const void * p, const void * q);

bool processBool(const char *arg);

#endif