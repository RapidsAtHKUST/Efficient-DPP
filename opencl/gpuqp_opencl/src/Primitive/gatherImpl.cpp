//
//  gatherImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "gatherImpl.h"

double gather(cl_mem d_source, cl_mem& d_dest, int length, cl_mem d_loc, int localSize, int gridSize, PlatInfo info) {
    
    double totalTime = 0;
    
    cl_int status = 0;
    int argsNum = 0;
    
    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/gatherKernel.cl");
    std::string kerAddr = path;
    
    char kerName[100] = "gatherKernel";
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel gatherKernel = reader.getKernel(kerName);
    
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_dest);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_loc);
    checkErr(status, ERR_SET_ARGUMENTS);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
    struct timeval start, end;
    
#ifdef PRINT_KERNEL
    printExecutingKernel(gatherKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, gatherKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    return totalTime;
}
