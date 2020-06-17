//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#pragma once

#include <iostream>

double compute_bandwidth(uint64_t num, int wordSize, double kernel_time);
double average_Hampel(double *input, int num);

void random_generator_int(int *keys, uint64_t length, int max, unsigned long long seed);
void random_generator_int_unique(int *keys, uint64_t length);