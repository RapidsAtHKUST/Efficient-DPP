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
#include <cstring>

//CUDA used header files
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#define MAX_DATA_SIZE 			(16000000)
#define MAX_NUM					(INT_MAX/2)
#define BLOCKSIZE 					(512)
#define GRIDSIZE					(1024)
#define SHUFFLE_TIME(TIME)		(TIME * 1.5)

typedef int2 Record;

int compRecordAsc ( const void * a, const void * b);
int compRecordDec ( const void * a, const void * b);
int compInt ( const void * p, const void * q);

//generate a sorted ascending record array
void recordSorted(Record *records, int length, int max=MAX_NUM);
void recordSorted_Only(Record *records, int length);
void recordRandom(Record *records, int length, int max=MAX_NUM);
void recordRandom_Only(Record *records, int length,  int times);
void intRandom(int *intArr, int length, int max=MAX_NUM);
void intRandom_Only(int *intArr, int length,  int times);

// void generateFixedRecords(Record* fixedRecords, int length, bool write, char *file);

// void generateFixedArray(int *fixedArray, int length, bool write, char *file);
void readFixedRecords(Record* fixedRecords, char *file, int& recordLength);
void readFixedArray(int* fixedArray, char *file, int & arrayLength);

double calCPUTime(clock_t start, clock_t end);
double diffTime(struct timeval end, struct timeval start);	//calculate the time

void printbinary(const unsigned int val, int dis);
int floorOfPower2_CPU(int a);

#endif