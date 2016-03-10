//
//  test.h
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#ifndef __TEST_H__
#define __TEST_H__

#include "kernels.h"

template<class T>
bool testMap(T *source, int r_len, float& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);

bool testGather(Record *source, int r_len, int *loc,double& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testGather_mul(Record *source, int r_len, int *loc,double& totalTime,  int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);

bool testScatter(Record *source, int r_len, int *loc,double& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testScan(int *source, int r_len, double& time,  int isExclusive, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testSplit(Record *source, int r_len, double& totalTime,  int fanout, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testRadixSort(Record *source, int r_len, double& totalTime, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testBisort(Record *source, int r_len, double& totalTime,int dir, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);

bool testRadixSort_int(int *source, int r_len, double& totalTime, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
#endif