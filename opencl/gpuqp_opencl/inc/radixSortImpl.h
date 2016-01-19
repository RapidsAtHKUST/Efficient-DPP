//
//  radixSortImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/6/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__radixSortImpl__
#define __gpuqp_opencl__radixSortImpl__

#include "Foundation.h"

double radixSort(cl_mem& d_source, int length, PlatInfo info);

#endif /* defined(__gpuqp_opencl__radixSortImpl__) */
