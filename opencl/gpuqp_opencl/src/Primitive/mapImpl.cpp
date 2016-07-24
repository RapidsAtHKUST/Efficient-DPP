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
double map(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem &d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    int localSize, int gridSize, PlatInfo info) 
{
    
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
    
#ifdef RECORDS
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_keys);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_keys);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(bool), &isRecord);
#endif
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(mapKernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, mapKernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1,&event);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    return totalTime;
}




   



