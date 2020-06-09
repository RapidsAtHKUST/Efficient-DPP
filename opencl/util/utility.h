//
//  utility.h
//  comparison_gpu
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#pragma once

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
double computeMem(unsigned long dataSize, int wordSize, double elapsedTime);
double averageHampel(double *input, int num);

