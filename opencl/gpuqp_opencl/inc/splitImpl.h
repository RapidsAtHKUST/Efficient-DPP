//
//  splitImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__splitImpl__
#define __gpuqp_opencl__splitImpl__

#include "Foundation.h"

double split(cl_mem d_source, cl_mem &d_dest, int length, int fanout, PlatInfo info, int localSize, int gridSize);

#endif /* defined(__gpuqp_opencl__splitImpl__) */
