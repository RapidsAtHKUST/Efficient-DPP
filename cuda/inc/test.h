//
//  test.h
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#ifndef __TEST_H__
#define __TEST_H__

#include "utility.h"
#include "kernels.h"

bool testMap(Record *source, int r_len, int blockSize, int gridSize);


#endif