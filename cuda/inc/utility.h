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

typedef int2 Record;

double diffTime(struct timeval end, struct timeval start);

#endif