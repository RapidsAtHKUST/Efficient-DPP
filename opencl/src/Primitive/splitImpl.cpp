//
//  splitImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Foundation.h"

/*
 *  Radix partitioning (A block of threads share a histogram)
 *  Input:  1.Table being partitioned,  (d_in)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *          4.Whether to use data reordering (reorder)
 *  Output: 1.Partitioned table         (d_out)
 *          2.Array recording the start position of each partition in the table (d_start)
 *
*/
double block_split_k(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, bool reorder, PlatInfo& info, int localSize, int gridSize, int sharedSize) {
//    std::cout<<"Function: block_split_k"<<std::endl;

//    localSize = 256, gridSize = 4096, sharedSize=8192;
    cl_int status = 0;
    cl_event event;

    int globalSize = localSize * gridSize;
    int argsNum = 0;
    double totalTime = 0;
    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    //kernel reading
    cl_kernel histogramKernel = KernelProcessor::getKernel("splitKernel.cl", "block_histogram", info.context);

    cl_kernel gatherHisKernel = KernelProcessor::getKernel("splitKernel.cl", "gatherStartPos", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    int his_len = buckets*gridSize;
    cl_mem d_his_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_his_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_his_in);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    status = clEnqueueNDRangeKernel(info.currentQueue, histogramKernel, 1, 0, global, local, 0, 0, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    status = clFinish(info.currentQueue);
    double histogramTime = clEventTime(event);
    totalTime += histogramTime;
    // std::cout<<"Histogram time: "<<histogramTime<<" ms."<<std::endl;

    //prefix scan
    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 64, 39, 112, 0);
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 1024, 15, 0, 11);
    totalTime += scanTime;
//    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;



    //gather the start position (optional)
//    argsNum = 0;
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_his);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &his_len);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_start);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &gridSize);
//    checkErr(status, ERR_SET_ARGUMENTS);
//
//    gettimeofday(&start, NULL);
//    status = clEnqueueNDRangeKernel(info.currentQueue, gatherHisKernel, 1, 0, global, local, 0, 0, 0);
//    status = clFinish(info.currentQueue);
//    gettimeofday(&end, NULL);
//    double gatherTime = diffTime(end, start);
//    std::cout<<"Gather time: "<<gatherTime<<" ms."<<std::endl;
//    totalTime += gatherTime;

    cl_kernel scatterWithHisKernel;
    if (reorder) {
        char extra[500] = {'\0'};
        if (buckets <= 8) {
            strcpy(extra, "-DSMALLER_WARP_SIZE");
        }    //one-level local scan
        else {  //two-level local scan
            int loop = (buckets+localSize-1)/localSize;             //number of elements each work-item processes
            if (loop == 1)  strcpy(extra, "-DLARGER_WARP_SIZE_SINGLE_LOOP");
            else {
                strcpy(extra, "-DLARGER_WARP_SIZE_MULTIPLE_LOOPS");
                char loop_str[20];
                my_itoa(loop, loop_str, 10);
                strcat(extra, " -DLOOPS=");
                strcat(extra, loop_str);
            }
        }
        scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "block_reorder_scatter_k",info.context,extra);
    }
    else scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "block_scatter_k", info.context);

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_out);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * (buckets+1), NULL);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &buckets);

    // int *h_help = new int[localSize];
    // int *h_help2 = new int[localSize];

    // cl_mem d_help = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* localSize, NULL, &status);
    // checkErr(status, ERR_HOST_ALLOCATION);
    // cl_mem d_help2 = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* localSize, NULL, &status);
    // checkErr(status, ERR_HOST_ALLOCATION);

    if (reorder) {
        status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_in);
        status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * sharedSize, NULL);
        // status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_help);
        // status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_help2);
    }
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    status = clEnqueueNDRangeKernel(info.currentQueue, scatterWithHisKernel, 1, 0, global, local, 0, 0, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    status = clFinish(info.currentQueue);
    double scatterTime = clEventTime(event);
    totalTime += scatterTime;
   // std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    // status = clEnqueueReadBuffer(info.currentQueue, d_help, CL_TRUE, 0, sizeof(int)*localSize, h_help, 0, 0, 0);
    // checkErr(status, ERR_WRITE_BUFFER);

    // status = clEnqueueReadBuffer(info.currentQueue, d_help2, CL_TRUE, 0, sizeof(int)*localSize, h_help2, 0, 0, 0);
    // checkErr(status, ERR_WRITE_BUFFER);

    // int total = 0;
    // std::cout<<"help:"<<std::endl;
    // for(int i = 0; i < localSize; i++) {
    //     std::cout<<h_help[i]<<' ';
    //     // if (i < 32)
    //     total += h_help[i];
    // }
    // std::cout<<std::endl;
    // std::cout<<"Total: "<<total<<std::endl;

    // total = 0;
    // std::cout<<"help2:"<<std::endl;
    // for(int i = 0; i < localSize; i++) {
    //     std::cout<<h_help2[i]<<' ';
    //     // total += h_help2[i];
    // }
    // std::cout<<std::endl;
    // // std::cout<<"Total: "<<total<<std::endl;

    // delete[] h_help;
    // delete[] h_help2;

    clReleaseMemObject(d_his_in);
    clReleaseMemObject(d_his_out);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}

double block_split_kv(cl_mem d_in_keys, cl_mem d_in_values, cl_mem d_out_keys, cl_mem d_out_values, cl_mem d_start, int length, int buckets, bool reorder, PlatInfo& info, int localSize, int gridSize, int sharedSize) {
//    std::cout<<"Function: block_split_kv"<<std::endl;

//    localSize = 128, gridSize = 65536, sharedSize=512;
    int globalSize = localSize * gridSize;
    cl_int status = 0;
    int argsNum = 0;
    double totalTime = 0;

    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    //kernel reading
    cl_kernel histogramKernel = KernelProcessor::getKernel("splitKernel.cl", "block_histogram", info.context);

    cl_kernel gatherHisKernel = KernelProcessor::getKernel("splitKernel.cl", "gatherStartPos", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    int his_len = buckets*gridSize;
    cl_mem d_his_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_his_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    struct timeval start, end;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_his_in);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, histogramKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    double histogramTime = diffTime(end, start);
//    std::cout<<"Histogram time: "<<histogramTime<<" ms."<<std::endl;

    totalTime += histogramTime;
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 1024, 15, 0, 11);
    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 64, 39, 112, 0);
    totalTime += scanTime;
//    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;


    //gather the start position (optional)
//    argsNum = 0;
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_his);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &his_len);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(cl_mem), &d_start);
//    status |= clSetKernelArg(gatherHisKernel, argsNum++, sizeof(int), &gridSize);
//    checkErr(status, ERR_SET_ARGUMENTS);
//
//    gettimeofday(&start, NULL);
//    status = clEnqueueNDRangeKernel(info.currentQueue, gatherHisKernel, 1, 0, global, local, 0, 0, 0);
//    status = clFinish(info.currentQueue);
//    gettimeofday(&end, NULL);
//    double gatherTime = diffTime(end, start);
//    std::cout<<"Gather time: "<<gatherTime<<" ms."<<std::endl;
//    totalTime += gatherTime;

    cl_kernel scatterWithHisKernel;
    if (reorder) {
        char extra[500] = {'\0'};
        if (buckets <= 32) strcpy(extra, "-DSMALLER_WARP_SIZE");    //one-level local scan
        else {  //two-level local scan
            int loop = (buckets+localSize-1)/localSize;             //number of elements each work-item processes
            if (loop == 1)  strcpy(extra, "-DLARGER_WARP_SIZE_SINGLE_LOOP");
            else {
                strcpy(extra, "-DLARGER_WARP_SIZE_MULTIPLE_LOOPS");
                char loop_str[20];
                my_itoa(loop, loop_str, 10);
                strcat(extra, " -DLOOPS=");
                strcat(extra, loop_str);
            }
        }
        scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "block_reorder_scatter_kv",info.context,extra);
    }
    else scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "block_scatter_kv", info.context);

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_values);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_values);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_out);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * (buckets+1), NULL);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &buckets);

    if (reorder) {
        status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_in);
        status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * sharedSize, NULL);
        status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int) * sharedSize, NULL);
    }

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
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    totalTime += scatterTime;

    clReleaseMemObject(d_his_in);
    clReleaseMemObject(d_his_out);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}

/*
 *  Thread-level partitioning (Each thread has a histogram)
 *  Input:  1.Table being partitioned,  (d_in_keys)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out_keys)
 *          2.Array recording the start position of each partition in the table (d_start)
 *
*/
double thread_split_k(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, PlatInfo& info, int localSize, int gridSize) {

    cl_event event;

//    localSize = 128, gridSize = 128;
    int globalSize = localSize * gridSize;
    cl_int status = 0;
    int argsNum = 0;
    double totalTime = 0;

    cl_kernel histogramKernel, scatterWithHisKernel;

    //kernel reading
    histogramKernel = KernelProcessor::getKernel("splitKernel.cl", "thread_histogram", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    unsigned long his_len = buckets*globalSize;       //larger size of histogram

    //check whether the histogram can be placed in the global memory (at most 2^32 Bytes)
//    long his_len_comp = his_len;
//    long limit = 1<<32;
//    if (his_len_comp*sizeof(int) >= limit)   return 9999;

    cl_mem d_his_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_his_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    struct timeval start, end;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_his_in);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogramKernel, argsNum++, localSize*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    status = clEnqueueNDRangeKernel(info.currentQueue, histogramKernel, 1, 0, global, local, 0, 0, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    status = clFinish(info.currentQueue);
    double histogramTime = clEventTime(event);
//    std::cout<<"Histogram time: "<<histogramTime<<" ms."<<std::endl;

    totalTime += histogramTime;
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 1024, 15, 0, 11);   //GPU
    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 64, 39, 110, 0);

    totalTime += scanTime;
//    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;

    scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "thread_scatter_k", info.context);

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_out);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, localSize*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    status = clEnqueueNDRangeKernel(info.currentQueue, scatterWithHisKernel, 1, 0, global, local, 0, 0, &event);
    status = clFinish(info.currentQueue);
    double scatterTime = clEventTime(event);

    //*2 for keys and values, another *2 for read and write data
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    totalTime += scatterTime;

    clReleaseMemObject(d_his_in);
    clReleaseMemObject(d_his_out);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}

double thread_split_kv(cl_mem d_in_keys, cl_mem d_in_values, cl_mem d_out_keys, cl_mem d_out_values, cl_mem d_start, int length, int buckets, PlatInfo& info, int localSize, int gridSize) {
//    std::cout<<"Function: thread_split_kv"<<std::endl;

//    localSize = 128, gridSize = 8192;
    int globalSize = localSize * gridSize;
    cl_int status = 0;
    int argsNum = 0;
    double totalTime = 0;

    cl_kernel histogramKernel, scatterWithHisKernel;

    //kernel reading
    histogramKernel = KernelProcessor::getKernel("splitKernel.cl", "thread_histogram", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)globalSize};

    unsigned long his_len = buckets*globalSize;       //larger size of histogram

    //check whether the histogram can be placed in the global memory (at most 2^32 Bytes)
//    long his_len_comp = his_len;
//    long limit = 1<<32;
//    if (his_len_comp*sizeof(int) >= limit)   return 9999;

    cl_mem d_his_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_his_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    struct timeval start, end;

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(cl_mem), &d_his_in);
    status |= clSetKernelArg(histogramKernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogramKernel, argsNum++, localSize*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(createListKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, histogramKernel, 1, 0, global, local, 0, 0, 0);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    double histogramTime = diffTime(end, start);
//    std::cout<<"Histogram time: "<<histogramTime<<" ms."<<std::endl;

    totalTime += histogramTime;
    checkErr(status, ERR_EXEC_KERNEL);

    //prefix scan
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 1024, 15, 0, 11);
    double scanTime = scan_fast(d_his_in, d_his_out, his_len, 1, info, 64, 39, 112, 0);

    totalTime += scanTime;
//    std::cout<<"Scan time:"<<scanTime<<" ms."<<std::endl;

    scatterWithHisKernel = KernelProcessor::getKernel("splitKernel.cl", "thread_scatter_kv", info.context);

    argsNum = 0;
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_in_values);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_out_values);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(cl_mem), &d_his_out);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatterWithHisKernel, argsNum++, localSize*buckets*sizeof(int), NULL);
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
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    totalTime += scatterTime;

    clReleaseMemObject(d_his_in);
    clReleaseMemObject(d_his_out);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}
