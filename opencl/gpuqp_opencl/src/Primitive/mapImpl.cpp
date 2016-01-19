//
//  mapImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "mapImpl.h"

/* Record *clSource
 * int * dest1
 * int * dest2
 */
double map(cl_mem d_source, int length, cl_mem& d_dest, int localSize, int gridSize, PlatInfo info) {
    
    double totalTime = 0;
    
    cl_int status = 0;
    int argsNum = 0;
    
    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/mapKernel.cl");
    std::string kerAddr = path;
    
    char kerName[100] = "mapKernel";
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel mapKernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    struct timeval start, end;
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(mapKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, mapKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);

    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    return totalTime;
}