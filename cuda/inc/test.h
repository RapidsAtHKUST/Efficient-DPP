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

bool testMap(Record *source, int r_len, double& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testGather(Record *source, int r_len, int *loc,double& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testScatter(Record *source, int r_len, int *loc,double& time, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);
bool testScan(int *source, int r_len, double& time,  int isExclusive, int blockSize=BLOCKSIZE, int gridSize=GRIDSIZE);

#endif