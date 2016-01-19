//
//  bisortImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/14/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__bisortImpl__
#define __gpuqp_opencl__bisortImpl__

#include <stdio.h>
#include "Foundation.h"

double bisort(cl_mem &d_source, int length, int dir, PlatInfo info, int localSize, int gridSize);
#endif /* defined(__gpuqp_opencl__bisortImpl__) */
