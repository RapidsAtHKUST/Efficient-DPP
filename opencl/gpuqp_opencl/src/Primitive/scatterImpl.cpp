//
//  scatterImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "scatterImpl.h"

double scatter(cl_mem d_source, cl_mem& d_dest, int length, cl_mem d_loc, int localSize, int gridSize, PlatInfo info) {
    
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;
    

    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/scatterKernel.cl");
    std::string kerAddr = path;
    
    char kerName[100] = "scatterKernel";
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel scatterKernel = reader.getKernel(kerName);
    
    int globalSize = localSize * gridSize;
    int ele_per_thread = (length + globalSize - 1) / (globalSize);
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_dest);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_loc);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &ele_per_thread);

    checkErr(status, ERR_SET_ARGUMENTS);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(scatterKernel);
#endif

    cl_event event;
    status = clEnqueueNDRangeKernel(info.currentQueue, scatterKernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    
    cl_ulong time_start, time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    totalTime = (time_end - time_start)/1000000.0;

    return totalTime;
}
