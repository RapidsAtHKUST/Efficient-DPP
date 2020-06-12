//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "../CL/cl.h"
#include "opencl_fake.h"
#include "openmp_fake.h"
#endif

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
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

/*literal macros*/
#define ERR_HOST_ALLOCATION                 "Failed to allocate the host memory."
#define ERR_WRITE_BUFFER                    "Failed to write to the buffer."
#define ERR_READ_BUFFER                     "Failed to read back the device memory."
#define ERR_SET_ARGUMENTS                   "Failed to set the arguments."
#define ERR_EXEC_KERNEL                     "Failed to execute the kernel."
#define ERR_LOCAL_MEM_OVERFLOW              "Local memory overflow "
#define ERR_COPY_BUFFER                     "Failed to copy the buffer."
#define ERR_RELEASE_MEM                     "Failed to release the device memory object."

#ifndef PROJECT_ROOT
#define PROJECT_ROOT " "
#endif

void checkErr(cl_int status, const char* name, int tag=-1);
void cl_mem_free(cl_mem object);
double clEventTime(const cl_event event);
void add_param(char *param, char *macro, bool has_value=false, int value=-1);

/*create the cl_kernel according to the file name and function name*/
cl_kernel get_kernel(cl_device_id device, cl_context context,
                     char *file_name, char *func_name, char *params=nullptr);

double diffTime(struct timeval end, struct timeval start);
void my_itoa(int num, char *buffer, int base);
double compute_bandwidth(unsigned long dataSize, int wordSize, double elapsedTime);
double average_Hampel(double *input, int num);

void random_generator_int(int *keys, uint64_t length, int max, unsigned long long seed);
void random_generator_int_unique(int *keys, uint64_t length);
