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
double split(cl_mem d_in_keys, cl_mem d_in_values, cl_mem d_out_keys, cl_mem d_out_values, cl_mem d_start, int length, int bits, PlatInfo& info) {

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
    cl_kernel histogramKernel = KernelProcessor::getKernel("splitKernel.cl", "histogram", info.context);
    cl_kernel scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "scatterWithHistogram", info.context);
    cl_kernel gatherHisKernel = KernelProcessor::getKernel("splitKernel.cl", "gatherStartPos", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    int his_len = buckets*gridSize;
    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    struct timeval start, end;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
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
    std::cout<<"Histogram time: "<<histogramTime<<" ms."<<std::endl;

    totalTime += histogramTime;
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
    double scanTime = scan_fast(d_his, his_len, 1, info, 1024, 15, 11, 0);
    totalTime += scanTime;
    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;

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
    std::cout<<"Gather time: "<<gatherTime<<" ms."<<std::endl;

    totalTime += gatherTime;

// -----------------  Test the size of each partition ------------------

//    int *h_start = new int[buckets];
//    status = clEnqueueReadBuffer(info.currentQueue, d_start, CL_TRUE, 0, sizeof(int)*buckets, h_start, 0, 0, 0);
//    checkErr(status, ERR_READ_BUFFER);
//    status = clFinish(info.currentQueue);
//
//    int largest = 0;
//    int smallest = 9999;
//    float ave = 0;
//
//    for(int i = 0; i < buckets; i++) {
//        int car;
//        if (i != buckets - 1)   car = h_start[i+1] - h_start[i];
//        else                    car = length - h_start[i];
//        if (car > largest)  largest = car;
//        if (car < smallest) smallest = car;
//        ave += car;
//    }
//    ave /= buckets;
//
//    std::cout<<"Partition results:"<<std::endl;
//    std::cout<<"\tBuckets: "<<buckets<<std::endl;
//    std::cout<<"\tLargest Part: "<<largest<<" ("<<largest* 2*sizeof(int)/1024<<" KB)"<<std::endl;
//    std::cout<<"\tSmallest Part: "<<smallest<<" ("<<smallest* 2*sizeof(int)/1024<<" KB)"<<std::endl;
//    std::cout<<"\tAverage Part: "<<ave<<" ("<<ave* 2*sizeof(int)/1024<<" KB)"<<std::endl;

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_values);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_values);
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

    //*2 for keys and values, another *2 for read and write data
    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    totalTime += scatterTime;

    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}