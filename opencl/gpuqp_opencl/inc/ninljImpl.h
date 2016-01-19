//
//  ninljImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/6/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__ninljImpl__
#define __gpuqp_opencl__ninljImpl__

#include "Foundation.h"

double ninlj(cl_mem& d_R, int rLen, cl_mem& d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize, int gridSize);

#endif /* defined(__gpuqp_opencl__ninljImpl__) */
