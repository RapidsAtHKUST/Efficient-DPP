//
//  utility.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/2016.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "utility.h"

double diffTime(struct timeval end, struct timeval start) {
	return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}