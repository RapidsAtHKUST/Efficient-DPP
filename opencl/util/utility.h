//
//  utility.h
//  comparison_gpu
//
//  Created by Zhuohang Lai on 01/19/16.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "../CL/cl.h"
#include "opencl_fake.h"
#include "openmp_fake.h"
#endif

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <fstream>
#include <climits>
#include <unistd.h>
#include <algorithm>
#include <assert.h>
#include <vector>

double diffTime(struct timeval end, struct timeval start);
void my_itoa(int num, char *buffer, int base);
double compute_bandwidth(unsigned long dataSize, int wordSize, double elapsedTime);
double average_Hampel(double *input, int num);

void random_generator_int(int *keys, uint64_t length, int max, unsigned long long seed);
void random_generator_int_unique(int *keys, uint64_t length);
