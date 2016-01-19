//
//  testJoins.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/21/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef gpuqp_opencl_testJoins_h
#define gpuqp_opencl_testJoins_h

#include "Foundation.h"

bool testNinlj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testInlj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testSmj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize = BLOCKSIZE);
bool testHj(int rLen, int sLen, PlatInfo info, int countBit, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);

#endif
