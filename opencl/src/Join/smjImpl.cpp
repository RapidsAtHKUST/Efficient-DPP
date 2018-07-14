//
//  smjImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/14/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double smj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize) {
    
    double totalTime = 0;
    
    int local_S_length = MAX_LOCAL_MEM_SIZE / sizeof(Record);       //5875
    int blockNum = ceil(1.0 * sLen / local_S_length);               //each block deals with local_S_length S records
    int itemNum = localSize * blockNum;
    
    cl_int status;
    int argsNum = 0;
    int totalResNum = 0;
    uint tempCount = 0;
    
    //kernel reading
    cl_kernel countMatchKernel = KernelProcessor::getKernel("smjKernel.cl", "countMatch", info.context);
    cl_kernel writeMatchKernel = KernelProcessor::getKernel("smjKernel.cl", "writeMatch", info.context);

    //memory allocation
    cl_mem d_count = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*itemNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * blockNum)};
    
    struct timeval start, end;
    
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(Record) * local_S_length, NULL);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &local_S_length);
    checkErr(status, ERR_SET_ARGUMENTS);
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(countMatchKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, countMatchKernel, 1, 0, global, mylocal, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    //get the last num
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE, sizeof(uint)*(itemNum-1), sizeof(uint), &tempCount, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    
    //scan
//    totalTime += scan(d_count, itemNum, 1, info);
    
    //get the last number
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE,sizeof(uint)*(itemNum-1), sizeof(uint), &tempCount, 0, NULL, NULL);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    oLen = totalResNum;
    
    if (totalResNum == 0) {
        return totalTime;
    }
    
    d_Out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*totalResNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //write count----------------------------------------------------------------------------------------------------------------
    
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_Out);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(Record) * local_S_length, NULL);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &local_S_length);
    checkErr(status, ERR_SET_ARGUMENTS);
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(writeMatchKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, writeMatchKernel, 1, 0, global, mylocal, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    status = clReleaseMemObject(d_count);
    checkErr(status, ERR_RELEASE_MEM);
    
    return  totalTime;
}