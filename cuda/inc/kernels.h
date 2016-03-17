//
//  kernels.h
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "utility.h"

//funcitioning on the host memory, mainly for testing
// extern "C" double gatherImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize);
// extern "C" double scatterImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize);
extern "C" double scanImpl(int *h_source, int r_len, int blockSize, int gridSize, int isExclusive);
extern "C" double splitImpl(Record *h_source, Record *h_dest, int r_len, int fanout, int blockSize, int gridSize);
extern "C" double radixSortImpl(Record *h_source, int r_len, int blockSize, int gridSize);
extern "C" double bitonicSortImpl(Record *h_source, int r_len, int dir, int blockSize, int gridSize);

extern "C" double radixSortImpl_int(int *h_source, int r_len, int blockSize, int gridSize);
// //for multi-path testing
// extern "C" double gatherImpl_mul(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize);

//directly functioning on the device memory



template<class T> float map(		
#ifdef RECORDS
	int *d_source_keys, int *d_des_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	) ;

template<class T> float gather(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template<class T> float gather_mul(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template<class T> float scatter(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template<class T> float scatter_mul(
#ifdef RECORDS
	int *d_source_keys, int *d_dest_keys,
#endif
	T *d_source_values, T *d_dest_values, 	
	int r_len, int* loc, int blockSize, int gridSize
#ifdef RECORDS
	,bool isRecord
#endif
	);

template<class T>
float split(
#ifdef RECORDS
    int *d_source_keys, int *d_dest_keys,
#endif
    T *d_source_values, T *d_dest_values, 
    int* d_his, int r_len, int fanout, int blockSize, int gridSize
#ifdef RECORDS
    ,bool isRecord
#endif
    );

template<typename T>
float scan(T *d_source, int length, int isExclusive, int blockSize);

// extern "C" double gatherDevice(Record *d_source, Record *d_res, int r_len,int *d_loc, int blockSize, int gridSize);
// extern "C" double scatterDevice(Record *d_source, Record *d_res, int r_len,int *d_loc, int blockSize, int gridSize);
extern "C" double scanDevice(int *d_source, int r_len, int blockSize, int gridSize, int isExclusive);
// extern "C" double splitDevice(Record *d_source, Record *d_dest, int* d_his, int r_len, int fanout, int blockSize, int gridSize);
extern "C" double radixSortDevice(Record *d_source, int r_len, int blockSize, int gridSize);
extern "C" double bitonicSortDevice(Record *d_source, int r_len, int dir, int blockSize, int gridSize);
extern "C" double bitonicSortDevice_op(Record *d_source, int r_len, int dir, int blockSize, int gridSize);

//currently used fastest scan

// extern "C" double scatterDevice_int(int *d_source, int *d_res, int r_len,int *d_loc, int blockSize, int gridSize);

extern  "C" void scan_warp_test();
extern  "C" void testRadixSort();
extern  "C" void scan_global(int *d_source, int length, int isExclusive, int blockSize);
#endif

