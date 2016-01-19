//
//  mapImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__mapImpl__
#define __gpuqp_opencl__mapImpl__

#include <stdio.h>
#include "Foundation.h"

double map(cl_mem d_source, int length, cl_mem& d_dest, int localSize, int gridSize, PlatInfo info);

#endif /* defined(__gpuqp_opencl__mapImpl__) */
