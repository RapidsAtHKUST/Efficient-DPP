//
//  barrierImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 23/08/16.
//  Copyright (c) 2016 Bryan. All rights reserved.
//

#include "Foundation.h"

#define BARRIER_EXPR_TIME       (10)
#define BARRIER_REPEAT_TIME     (1000)


double my_barrier(
    cl_mem d_source_values, int localSize, int gridSize, PlatInfo& info, double& percentage)
{
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/barrierKernel.cl");
    std::string kerAddr = path;

    char kerName_in[100] = "barrier_in";
    char kerName_free[100] = "barrier_free";

    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel kernel_in = reader.getKernel(kerName_in);
    cl_kernel kernel_free = reader.getKernel(kerName_free);

    const int repeatTime = BARRIER_REPEAT_TIME;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel_in, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel_in, argsNum++, sizeof(int), &repeatTime);
    checkErr(status, ERR_SET_ARGUMENTS);

    argsNum = 0;
    status |= clSetKernelArg(kernel_free, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel_free, argsNum++, sizeof(int), &repeatTime);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel_in);
    printExecutingKernel(kernel_free);
#endif

    double totalTime_in = 0.0;
    double totalTime_free = 0.0;

    cl_event event;
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    for(int i = 0; i < BARRIER_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_in, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_in = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_in += tempTime_in;

        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_free, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_free = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_free += tempTime_free;
    }    
    totalTime_in /= (BARRIER_EXPR_TIME - 1);
    totalTime_free /= (BARRIER_EXPR_TIME - 1);

    totalTime = totalTime_in - totalTime_free;
    percentage = totalTime / totalTime_in * 100;

    return totalTime;
}





   



