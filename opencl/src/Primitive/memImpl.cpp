//
//  mapImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"


double mem_read(
    cl_mem d_source_values, cl_mem d_dest_values, int length, 
    int localSize, int gridSize, PlatInfo info, int con, int basicSize)
{
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;
    my_itoa(basicSize, basicSizeName, 10);
    char kerName[100] = "mem_read_float";
    strcat(kerName, basicSizeName);
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel kernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &length);

    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, kernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1,&event);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);
    

    return totalTime;
}

double mem_write(
    cl_mem d_source_values, int length, 
    int localSize, int gridSize, PlatInfo info, int con, int basicSize)
{
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;
    my_itoa(basicSize, basicSizeName, 10);
    char kerName[100] = "mem_write_float";
    strcat(kerName, basicSizeName);
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel kernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &length);
    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, kernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1,&event);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    return totalTime;
}

double triad(
    cl_mem d_source_values_b, cl_mem d_source_values_c, cl_mem d_dest_values_a,int length, int localSize, int gridSize, PlatInfo info, int basicSize)
{
    double totalTime = 0;

    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;
    my_itoa(basicSize, basicSizeName, 10);
    char kerName[100] = "triad_float";
    strcat(kerName, basicSizeName);
    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel kernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_dest_values_a);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values_b);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values_c);
    // status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &length);
    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, kernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    clWaitForEvents(1,&event);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    std::cout<<"tempTime: "<<totalTime<<" ms."<<std::endl;
    return totalTime;
}



   



