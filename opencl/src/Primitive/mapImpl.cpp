//
//  mapImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double map_hashing(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat) 
{
    double totalTime = 0;
    cl_int status = 0;
    
    //kernel reading
//    char path[100] = PROJECT_ROOT;
//    strcat(path, "/Kernels/mapKernel.cl");
//    std::string kerAddr = path;
//
//    char kerName[100] = "map_hash";
//    KernelProcessor reader(&kerAddr,1,info.context);
//    cl_kernel mapKernel = reader.getKernel(kerName);

    cl_kernel mapKernel = KernelProcessor::getKernel("mapKernel.cl", "map_hash", info.context);

    //set kernel arguments
    int argsNum = 0;
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(int), &repeat);
    
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
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    return totalTime;
}

double map_branching(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat,int branch)  
{
    double totalTime = 0;
    cl_int status = 0;
    
    //kernel reading
    cl_kernel b1_Kernel = KernelProcessor::getKernel("mapKernel.cl", "map_branch_", info.context);

    //set kernel arguments for b1
    int argsNum = 0;
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(int), &repeat);

    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(b1_Kernel);
#endif
    //b1
    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, b1_Kernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    return totalTime;
}

double map_branching_for(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat,int branch)  
{
    double totalTime = 0;
    cl_int status = 0;
    
    //kernel reading
    cl_kernel b1_Kernel = KernelProcessor::getKernel("mapKernel.cl", "map_branch_for", info.context);

    //set kernel arguments for b1
    int argsNum = 0;
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(int), &repeat);
    status |= clSetKernelArg(b1_Kernel, argsNum++, sizeof(int), &branch);
    
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(b1_Kernel);
#endif
    //b1
    cl_event event;
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, b1_Kernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    return totalTime;
}

template<typename T>
void map_transform(
    cl_mem d_source_alpha, cl_mem& d_source_beta, T r, 
    cl_mem &d_dest_x, cl_mem& d_dest_y, cl_mem& d_dest_z,
    int localSize, int gridSize, PlatInfo& info, int repeat, 
    double &blank_time, double &total_time) 
{
    cl_int status = 0;
    
    //kernel reading
    char kerName[100] = "map_trans";
    char blank_kerName[100] = "map_trans_blank";

    if (sizeof(T) == sizeof(float)) {
        strcat(kerName, "_float");
        strcat(blank_kerName, "_float");
    }
    else if (sizeof(T) == sizeof(double))   {
        strcat(kerName, "_double");
        strcat(blank_kerName, "_double");
    }

    cl_kernel mapKernel = KernelProcessor::getKernel("mapKernel.cl", kerName, info.context);
    cl_kernel blank_mapKernel = KernelProcessor::getKernel("mapKernel.cl", blank_kerName, info.context);

    //set kernel arguments
    int argsNum = 0;
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_alpha);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_source_beta);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(T), &r);

    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_x);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_y);
    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(cl_mem), &d_dest_z);

    status |= clSetKernelArg(mapKernel, argsNum++, sizeof(int), &repeat);

    argsNum = 0;
    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(cl_mem), &d_source_alpha);
    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(cl_mem), &d_source_beta);
    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(T), &r);

    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(cl_mem), &d_dest_x);
    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(cl_mem), &d_dest_y);
    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(cl_mem), &d_dest_z);

    status |= clSetKernelArg(blank_mapKernel, argsNum++, sizeof(int), &repeat);


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
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);
    total_time = clEventTime(event);

    status = clEnqueueNDRangeKernel(info.currentQueue, blank_mapKernel, 1, 0, global, local, 0, 0, &event);
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    checkErr(status, ERR_EXEC_KERNEL);
    blank_time = clEventTime(event);
}

template void map_transform<float>(
    cl_mem d_source_alpha, cl_mem& d_source_beta, float r, 
    cl_mem &d_dest_x, cl_mem& d_dest_y, cl_mem& d_dest_z,
    int localSize, int gridSize, PlatInfo& info, int repeat, double &blank_time, double &total_time);

template void map_transform<double>(
    cl_mem d_source_alpha, cl_mem& d_source_beta, double r, 
    cl_mem &d_dest_x, cl_mem& d_dest_y, cl_mem& d_dest_z,
    int localSize, int gridSize, PlatInfo& info, int repeat, double &blank_time, double &total_time);



   



