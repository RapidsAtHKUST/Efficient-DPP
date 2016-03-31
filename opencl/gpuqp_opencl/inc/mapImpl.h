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

double map(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    int localSize, int gridSize, PlatInfo info) ;

#endif /* defined(__gpuqp_opencl__mapImpl__) */
