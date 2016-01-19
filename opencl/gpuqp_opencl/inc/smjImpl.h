//
//  smjImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/11/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__smjImpl__
#define __gpuqp_opencl__smjImpl__

#include "Foundation.h"

double smj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize);

#endif /* defined(__gpuqp_opencl__smjImpl__) */
