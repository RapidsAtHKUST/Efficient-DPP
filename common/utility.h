//
//  utility.h
//  OpenCL-Primitives
//
//  Created by Zhuohang Lai on 01/19/16.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#ifndef __UTILITY_H__
#define __UTILITY_H__

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
double compute_bandwidth(uint64_t dataSize, int wordSize, double elapsedTime);
double average_Hampel(double *input, int num);

#endif