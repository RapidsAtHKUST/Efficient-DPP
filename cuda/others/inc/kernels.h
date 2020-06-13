//
//  kernels.h
//  gpuqp_cuda
//
//  Created by Zhuohang Lai on 01/19/16.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#ifndef __KERNELS_H__
#define __KERNELS_H__


#include "utility.h"

float mul(int *d_in, int *d_out, int blockSize, int gridSize);

//template<class T> float split(
//#ifdef RECORDS
//    int *d_source_keys, int *d_dest_keys,
//#endif
//    T *d_source_values, T *d_dest_values,
//    int* d_his, int r_len, int fanout, int blockSize, int gridSize
//#ifdef RECORDS
//    ,bool isRecord
//#endif
//    );

//template<typename T>
//float scan_warpwise(T *d_source, int length, int isExclusive, int blockSize);
//
//template<typename T>
//float scan_ble(T *d_source, int length, int isExclusive, int blockSize);


//template<typename T>
//float scan(T *d_source, int length, int isExclusive, int blockSize);

//template<typename T> float radixSort(
//#ifdef RECORDS
//    int *d_source_keys,
//#endif
//    T *d_source_values, int len
//#ifdef RECORDS
//    ,bool isRecord
//#endif
//    );

//deprecated
//DEPRECATED  double scanDevice(int *d_source, int r_len, int blockSize, int gridSize, int isExclusive);
//DEPRECATED  double radixSortDevice(Record *d_source, int r_len, int blockSize, int gridSize);
//DEPRECATED  double bitonicSortDevice(Record *d_source, int r_len, int dir, int blockSize, int gridSize);
//DEPRECATED  double bitonicSortDevice_op(Record *d_source, int r_len, int dir, int blockSize, int gridSize);
//DEPRECATED  double scanImpl(int *h_source, int r_len, int blockSize, int gridSize, int isExclusive);
//DEPRECATED  double splitImpl(Record *h_source, Record *h_dest, int r_len, int fanout, int blockSize, int gridSize);
//DEPRECATED  double radixSortImpl(Record *h_source, int r_len, int blockSize, int gridSize);
//DEPRECATED  double bitonicSortImpl(Record *h_source, int r_len, int dir, int blockSize, int gridSize);
//DEPRECATED  double radixSortImpl_int(int *h_source, int r_len, int blockSize, int gridSize);

#endif

