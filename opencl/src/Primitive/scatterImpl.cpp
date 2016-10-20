//
//  scatterImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double scatter(cl_mem d_source_values, cl_mem& d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, const PlatInfo info, int numOfRun) {
    
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
    
    //set kernel arguments
    argsNum = 0;

    int globalSize = gridSize * localSize;
    int ele_per_thread = (length + globalSize - 1) / globalSize;

    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_loc);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &ele_per_thread);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &numOfRun);

    checkErr(status, ERR_SET_ARGUMENTS);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
    
#ifdef PRINT_KERNEL
    printExecutingKernel(scatterKernel);
#endif

    status = clFinish(info.currentQueue);

    cl_event event;
    status = clEnqueueNDRangeKernel(info.currentQueue, scatterKernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);

    totalTime = clEventTime(event);

    return totalTime;
}
