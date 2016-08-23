//
//  mapImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

#define VPU_EXPR_TIME     (2)

/* basicSize:
 *  1 - float
 *  2 - float2
 *  3 - float3
 *  4 - float4
 *  8 - float8
 *  16 - float16
 *
 */
double vpu(
    cl_mem d_source_values, int length, 
    int localSize, int gridSize, PlatInfo& info, int con, int repeatTime, int basicSize)
{
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/vpuKernel.cl");
    std::string kerAddr = path;
    
    char kerName[100] = "vpu";
    char basicSizeName[20] = "";
    my_itoa(basicSize, basicSizeName, 10);
    strcat(kerName, basicSizeName);
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel vpuKernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(int), &con);
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(int), &repeatTime);
    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(vpuKernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);

    for(int i = 0; i < VPU_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, vpuKernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime += tempTime;
    }    
    totalTime /= (VPU_EXPR_TIME - 1);

    return totalTime;
}




   



