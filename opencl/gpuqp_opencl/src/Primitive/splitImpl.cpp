//
//  splitImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "splitImpl.h"
#include "scanImpl.h"

double split(cl_mem d_source, cl_mem &d_dest, int length, int fanout, PlatInfo info, int localSize, int gridSize) {
    
    double totalTime = 0;
    
    checkLocalMemOverflow(sizeof(int) * fanout * localSize);
    
    int globalSize = localSize * gridSize;
    
    cl_int status = 0;
    int argsNum = 0;
    
    //kernel reading
    char splitPath[100] = PROJECT_ROOT;
    strcat(splitPath, "/Kernels/splitKernel.cl");
    std::string splitkerAddr = splitPath;
    
    char createListSource[100] = "createList";
    char splitWithListSource[100] = "splitWithList";
    
    KernelProcessor splitReader(&splitkerAddr,1,info.context);
    cl_kernel createListKernel = splitReader.getKernel(createListSource);
    cl_kernel splitWithListKernel = splitReader.getKernel(splitWithListSource);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};
    
    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* fanout * globalSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    struct timeval start, end;
    
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(createListKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(createListKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(createListKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(createListKernel, argsNum++, sizeof(int) * fanout * localSize, NULL);
    status |= clSetKernelArg(createListKernel, argsNum++, sizeof(int), &fanout);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, createListKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
    totalTime += scan(d_his, fanout * globalSize , 1, info);
    
    argsNum = 0;
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(cl_mem), &d_dest);
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(int) * fanout * localSize, NULL);
    status |= clSetKernelArg(splitWithListKernel, argsNum++, sizeof(int), &fanout);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, splitWithListKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    status = clReleaseMemObject(d_his);
    checkErr(status, ERR_RELEASE_MEM);
    
    return totalTime;
}