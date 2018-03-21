//
//  splitImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

/*
 *  Radix partitioning
 *  Input:  1.Table being partitioned,  (d_in)
 *          2.Table cadinality,         (length)
 *          3.Reference bit num         (bits)
 *  Output: 1.Partitioned table         (d_out)
 *          2.Array recording the start position of each partition in the table (d_start)
 *
 */
double split(cl_mem d_in, cl_mem d_out, cl_mem d_start, int length, int bits, PlatInfo& info) {

    int localSize = 1024, gridSize = 1024;
    int globalSize = localSize * gridSize;
    cl_int status = 0;
    int argsNum = 0;
    double totalTime = 0;
    int buckets = (1<<bits);

//    SHOW_PARALLEL(localSize, gridSize);
//    SHOW_DATA_NUM(length);
    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    //kernel reading
    char splitPath[100] = PROJECT_ROOT;
    strcat(splitPath, "/Kernels/splitKernel.cl");
    std::string splitkerAddr = splitPath;
    
    char histogramSource[100] = "histogram";
    char scatterWithHisSource[100] = "scatterWithHistogram";
    char gatherHisSource[100] = "gatherStartPos";

    KernelProcessor splitReader(&splitkerAddr,1,info.context);
    cl_kernel histogramKernel = splitReader.getKernel(histogramSource);
    cl_kernel scatterWithHisKernel = splitReader.getKernel(scatterWithHisSource);
    cl_kernel gatherHisKernel = splitReader.getKernel(gatherHisSource);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    int his_len = buckets*gridSize;
    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    struct timeval start, end;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(short), &bits);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, histogramKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    double histogramTime = diffTime(end, start);
//    std::cout<<"Histogram time: "<<tempTime<<" ms."<<std::endl;

    totalTime += histogramTime;
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
    double scanTime = scan_fast(d_his, his_len, 1, info, 1024, 15, 11, 0);
    totalTime += scanTime;
//    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;

    //gather the start position
    argsNum = 0;
    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &his_len);
    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_start);
    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &gridSize);
    checkErr(status, ERR_SET_ARGUMENTS);

    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, gatherHisKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    double gatherTime = diffTime(end, start);
//    std::cout<<"Gather time: "<<tempTime<<" ms."<<std::endl;

    totalTime += gatherTime;

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(short), &bits);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, scatterWithHisKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    double scatterTime = diffTime(end, start);
//    std::cout<<"Scatter time: "<<tempTime<<" ms."<<std::endl;

    totalTime += scatterTime;

    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}