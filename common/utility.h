//
//  utility.h
//  comparison_gpu
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
#include <fstream>
#include <climits>
#include <unistd.h>
#include <algorithm>
#include <assert.h>

#ifdef CUDA_PROJ
	//CUDA used header files
	#include <cuda_runtime.h>
	#include <helper_cuda.h>
	#include <thrust/scan.h>
	#include <thrust/device_ptr.h>
	#include <thrust/device_vector.h>
	#include <thrust/sort.h>

	typedef int2 Record;
#endif

#ifdef OPENCL_PROJ
	#if defined(__APPLE__) || defined(__MACOSX)
		#include <OpenCL/OpenCL.h>
	#else
		#include <CL/cl.h>
	#endif
	
	#ifdef KERNEL
    	typedef int2 Record;
	#else
    	typedef cl_int2 Record;
	#endif

    //OpenCL error checking functions
	void checkErr(cl_int status, const char* name);
	double clEventTime(cl_event);
#endif

#ifdef OPENMP_PROJ
	#include <omp.h>
	#include <immintrin.h>
	typedef struct Record {
		int x;
		int y;
		Record(){}
		Record(int x, int y) {
			this->x = x;
			this->y = y;
		}
	} Record;
#endif

#ifdef __GNUC__
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

#define MAX_DATA_SIZE 			(160000000)
#define MAX_NUM					(INT_MAX/2)
#define BLOCKSIZE 					(256)
#define GRIDSIZE					(1024)
#define SHUFFLE_TIME(TIME)		(TIME * 1.5)
#define SHARED_MEM_SIZE			(48000)
#define SHARED_MEM_CHECK(SIZE)		assert(SIZE <= SHARED_MEM_SIZE);
#define MAX_TIME 				(99999.0f)

int compRecordAsc ( const void * a, const void * b);
int compRecordDec ( const void * a, const void * b);
int compInt ( const void * p, const void * q);

//generate a sorted ascending record array
template<typename T>
void recordSorted(int *keys, T *values, int length, T max= MAX_NUM);

template<typename T>
void recordSorted_Only(int *keys, T *values, int length);

template<typename T>
void recordRandom(int *keys, T *values, int length, T max=MAX_NUM);

template<typename T>
void recordRandom_Only(int *keys, T *values, int length, int times);

template<typename T>
void valRandom(T *arr, int length, T max=MAX_NUM);

template<typename T>
void valRandom_Only(T *arr, int length,  int times);

// void generateFixedRecords(Record* fixedRecords, int length, bool write, char *file);

// void generateFixedArray(int *fixedArray, int length, bool write, char *file);
void readFixedRecords(Record* fixedRecords, char *file, int& recordLength);
void readFixedArray(int* fixedArray, char *file, int & arrayLength);

double calCPUTime(clock_t start, clock_t end);
double diffTime(struct timeval end, struct timeval start);	//calculate

int floorOfPower2_CPU(int a);
void printRes(std::string funcName, bool res, double elaspsedTime);
void printRes(std::string funcName, bool res, float elaspsedTime);

#endif
