//
//  kernels.h
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "utility.h"

extern "C" void mapImpl(Record *h_source, Record *h_res, int r_len, int blockSize, int gridSize, double& time) ;
extern "C" void gatherImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize, double& time);

extern "C" void scatterImpl(Record *h_source, Record *h_res, int r_len,int *h_loc, int blockSize, int gridSize, double& time);
#endif