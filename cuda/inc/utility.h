//
//  utility.h
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#ifndef __UTILITY_H__
#define __UTILITY_H__

//cpp used header files
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <sys/time.h>

//CUDA used header files
#include <cuda_runtime.h>
#include <helper_cuda.h>


#define BLOCKSIZE 	(512)
#define GRIDSIZE	(1024)

typedef int2 Record;

int compRecordAsc ( const void * a, const void * b);
int compRecordDec ( const void * a, const void * b);
int compInt ( const void * p, const void * q);

//generate a sorted ascending record array
void recordSorted(Record *records, int length, int max=10000000);
void recordSorted_Only(Record *records, int length);
void recordRandom(Record *records, int length, int max=10000000);
void recordRandom_Only(Record *records, int length,  int times);
void intRandom(int *intArr, int length, int max=10000000);
void intRandom_Only(int *intArr, int length,  int times);

// void generateFixedRecords(Record* fixedRecords, int length, bool write, char *file);

// void generateFixedArray(int *fixedArray, int length, bool write, char *file);
// void readFixedRecords(Record* fixedRecords, char *file, int& recordLength);
// void readFixedArray(int* fixedArray, char *file, int & arrayLength);

double calCPUTime(clock_t start, clock_t end);
double diffTime(struct timeval end, struct timeval start);	//calculate the time

void printbinary(const unsigned int val, int dis);
int floorOfPower2_CPU(int a);

#endif