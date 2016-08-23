//
//  bisortImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/14/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

int getNearestLarger2Power(int input) {
    int k = 1;
    while (k < input)   k<<=1;
    return k;
}

//dir: 1 for asc, 0 for des
double bisort(cl_mem &d_source, int length, int dir, PlatInfo& info, int localSize, int gridSize) {
    
    double totalTime = 0;
    
    cl_int status = 0;
    int ceil = getNearestLarger2Power(length);      //get the padded length
    
    Record *maxArr = NULL;
    if (length != ceil) {
        maxArr = new Record[ceil-length];
        for(int i = 0; i < ceil - length; i++) {
            maxArr[i].x = INT_MAX;
            maxArr[i].y = INT_MAX;
        }
    }
    
    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/bitonicSortKernel.cl");
    std::string kerAddr = path;

    char kerName[100] = "bitonicSort";
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel bisortKernel = reader.getKernel(kerName);
    
    //memory allocation
    cl_mem d_temp = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(Record)*ceil, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueCopyBuffer(info.currentQueue, d_source, d_temp, 0, 0, sizeof(Record)*length, 0, 0, 0);
    checkErr(status, ERR_COPY_BUFFER);
    
    if (ceil != length) {
        status = clEnqueueWriteBuffer(info.currentQueue, d_temp, CL_TRUE, sizeof(Record)*length, sizeof(Record)*(ceil-length), maxArr , 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
    }
    
    //set kernel arguments
    status |= clSetKernelArg(bisortKernel, 0, sizeof(cl_mem), &d_temp);
    status |= clSetKernelArg(bisortKernel, 1, sizeof(int), &ceil);
//    status |= clSetKernelArg(bisortKernel, 3, sizeof(int), &dir);
    checkErr(status, ERR_SET_ARGUMENTS);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    struct timeval start, end;
    
    //bitonic sort
    for(int base = 2; base <= ceil ; base <<= 1) {
        for(int interval = base>>1 ; interval > 0; interval>>= 1) {
            
            status |= clSetKernelArg(bisortKernel, 2, sizeof(int), &base);
            status |= clSetKernelArg(bisortKernel, 3, sizeof(int), &interval);
            checkErr(status, ERR_SET_ARGUMENTS);
            
#ifdef PRINT_KERNEL
            printExecutingKernel(bisortKernel);
#endif
           gettimeofday(&start, NULL);
            status = clEnqueueNDRangeKernel(info.currentQueue, bisortKernel, 1, 0, global, local, 0, 0, 0);
            status = clFinish(info.currentQueue);
            gettimeofday(&end, NULL);
            totalTime += diffTime(end, start);
            checkErr(status, ERR_EXEC_KERNEL);
        }
    }
    
    //memory written back
    if (dir == 0)  { //descending
        status = clEnqueueCopyBuffer(info.currentQueue, d_temp, d_source, sizeof(Record)*(ceil-length), 0, sizeof(Record)*length, 0, 0, 0);
        checkErr(status, ERR_COPY_BUFFER);
    }
    else {          //ascending
        status = clEnqueueCopyBuffer(info.currentQueue, d_temp, d_source, 0, 0, sizeof(Record)*length, 0, 0, 0 );
        checkErr(status, ERR_COPY_BUFFER);
    }

    if (length != ceil)    delete [] maxArr;
    
    status = clReleaseMemObject(d_temp);
    checkErr(status, ERR_RELEASE_MEM);
    
    return totalTime;
}