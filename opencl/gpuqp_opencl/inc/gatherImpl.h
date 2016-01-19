//
//  gatherImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__gatherImpl__
#define __gpuqp_opencl__gatherImpl__

#include "Foundation.h"

double gather(cl_mem d_source, cl_mem& d_dest, int length, cl_mem d_loc, int localSize, int gridSize, PlatInfo info);

#endif /* defined(__gpuqp_opencl__gatherImpl__) */
