//
//  gatherImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "gatherImpl.h"

double gather(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem &d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    cl_mem d_loc, int localSize, int gridSize, PlatInfo info) {
    
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

    int globalSize = gridSize * localSize;
    int ele_per_thread = (length + globalSize - 1) / globalSize;

#ifdef RECORDS
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_keys);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_keys);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(bool), &isRecord);
#endif
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_loc);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(int), &ele_per_thread);
    checkErr(status, ERR_SET_ARGUMENTS);
    
    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
    
#ifdef PRINT_KERNEL
    printExecutingKernel(gatherKernel);
#endif

    status = clFinish(info.currentQueue);

    cl_event event;
    status = clEnqueueNDRangeKernel(info.currentQueue, gatherKernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1, &event);
    checkErr(status, ERR_EXEC_KERNEL);

    totalTime = clEventTime(event);

    return totalTime;
}
