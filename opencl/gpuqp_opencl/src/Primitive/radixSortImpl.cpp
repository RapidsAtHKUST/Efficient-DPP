//
//  radixSortImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/6/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "radixSortImpl.h"
#include "scanImpl.h"
#include "scatterImpl.h"

double radixSort(cl_mem& d_source, int length, PlatInfo info) {
    
    double totalTime = 0;
    
    int blockSize = 47;             //local memory size: 47KB
    int gridSize = 64;
    
    cl_int status;
    int argsNum = 0;

    int bits = 8;                   //each pass sort 8 bits
    int radix = (1<<bits);
    
    uint hisSize = blockSize * gridSize * radix;
    uint isExclusive = 1;
    
    //kernel reading
    char sortPath[100] = PROJECT_ROOT;
    strcat(sortPath, "/Kernels/radixSortKernel.cl");
    std::string sortKerAddr = sortPath;
    
    char countHisSource[100] = "countHis";
    char writeHisSource[100] = "writeHis";
    
    KernelProcessor sortReader(&sortKerAddr,1,info.context);
    
    cl_kernel countHisKernel = sortReader.getKernel(countHisSource);
    cl_kernel writeHisKernel = sortReader.getKernel(writeHisSource);
    
    size_t testLocal[1] = {(size_t)blockSize};
    size_t testGlobal[1] = {(size_t)(blockSize * gridSize)};
    
    struct timeval start, end;
    
    cl_mem d_temp = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_histogram = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)* hisSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    for(uint shiftBits = 0; shiftBits < sizeof(int) * 8; shiftBits += bits) {
        
        isExclusive = 1;        //reset the exclusive
        //data preparation

        argsNum = 0;
        status |= clSetKernelArg(countHisKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(countHisKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(countHisKernel, argsNum++, sizeof(cl_mem), &d_histogram);
        status |= clSetKernelArg(countHisKernel, argsNum++, sizeof(ushort)*radix*blockSize, NULL);
        status |= clSetKernelArg(countHisKernel, argsNum++, sizeof(uint), &shiftBits);
        checkErr(status, ERR_SET_ARGUMENTS);
        
#ifdef PRINT_KERNEL
        printExecutingKernel(countHisKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, countHisKernel, 1, 0, testGlobal, testLocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
        
        totalTime += scan(d_histogram,hisSize,1,info);
        
        argsNum = 0;
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(uint), &length);
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(cl_mem), &d_histogram);
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(cl_mem), &d_loc);
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(uint)*radix*blockSize, NULL);
        status |= clSetKernelArg(writeHisKernel, argsNum++, sizeof(uint), &shiftBits);
        checkErr(status, ERR_SET_ARGUMENTS);
        
#ifdef PRINT_KERNEL
        printExecutingKernel(writeHisKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, writeHisKernel, 1, 0, testGlobal, testLocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
        
        //scatter
        totalTime += scatter(d_source, d_temp, length, d_loc, 512, 32768, info);
        
        //copy the buffer for another loop
        status = clEnqueueCopyBuffer(info.currentQueue, d_temp, d_source, 0, 0, sizeof(Record)*length,0 , 0, 0);
        checkErr(status, ERR_COPY_BUFFER);
    }
    
    status = clReleaseMemObject(d_temp);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_histogram);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);

    return totalTime;
}
