//
//  scanImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__scanImpl__
#define __gpuqp_opencl__scanImpl__

#include "Foundation.h"

double scan(cl_mem &cl_arr, int num,int isExclusive, PlatInfo info, int localSize = BLOCKSIZE);


#endif /* defined(__gpuqp_opencl__scanImpl__) */
