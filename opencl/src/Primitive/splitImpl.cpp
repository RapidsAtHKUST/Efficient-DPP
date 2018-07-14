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
 *  Input:  1.Table being partitioned,  (d_in_keys, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *          4.Whether to use data reordering (reorder)
 *  Output: 1.Partitioned table         (d_out_keys, d_out_values)
 *          2.Array recording the start position of each partition in the table (d_start)
 *
*/
double block_split(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, bool reorder, PlatInfo& info, cl_mem d_in_values, cl_mem d_out_values, int local_size, int grid_size) {

    local_size = 64, grid_size = 16384;

    //check data type
    bool key_only;
    if ((d_in_values != NULL) && (d_out_values != NULL)) key_only = false;
    else if ((d_in_values == NULL) && (d_out_values == NULL)) key_only = true;
    else {
        std::cerr << "Wrong parameters." << std::endl;
        return -1;
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    cl_kernel histogram_kernel, scatter_kernel, gather_his_kernel;
    cl_mem d_his, d_his_origin;
    double histogram_time, gather_time, scan_time, scatter_time, total_time = 0;

    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t) local_size};
    size_t global[1] = {(size_t) global_size};

/*1.histogram*/
    histogram_kernel = KernelProcessor::getKernel("splitKernel.cl", "block_histogram", info.context);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(info.currentQueue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

    //copy the global histogram before scan
    if (reorder) {
        d_his_origin = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        status = clEnqueueCopyBuffer(info.currentQueue, d_his, d_his_origin, 0, 0, sizeof(int) * his_len, 0, 0, 0);
        checkErr(status, ERR_EXEC_KERNEL);
        status = clFinish(info.currentQueue);
    }

/*2.scan*/
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len,  info, 1024, 15, 0, 11);
    scan_time = scan_fast(d_his, his_len, info, 64, 39, 112, 0);
    total_time += scan_time;

/*2.5 gather the start position (optional)*/
    if (d_start != NULL) {
        gather_his_kernel = KernelProcessor::getKernel("splitKernel.cl", "gatherStartPos", info.context);
        argsNum = 0;
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &his_len);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_start);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &grid_size);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clEnqueueNDRangeKernel(info.currentQueue, gather_his_kernel, 1, 0, global, local, 0, 0, &event);
        status = clFinish(info.currentQueue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

/*3.scatter*/
    //compilation parameters
    char para_s[500] = {'\0'};
    if (!key_only)  strcat(para_s, "-DKVS ");
    if (reorder) {
        if (buckets <= 32) strcat(para_s, "-DSMALLER_WARP_SIZE ");   //one-level local scan
        else {  //two-level local scan
            int loop = (buckets+local_size-1)/local_size;          //number of elements each work-item processes
            if (loop == 1)  strcat(para_s, "-DLARGER_WARP_SIZE_SINGLE_LOOP ");
            else {
                strcat(para_s, "-DLARGER_WARP_SIZE_MULTIPLE_LOOPS");
                char loop_str[20];
                my_itoa(loop, loop_str, 10);
                strcat(para_s, " -DLOOPS=");
                strcat(para_s, loop_str);
            }
        }
        scatter_kernel = KernelProcessor::getKernel("splitKernel.cl", "block_reorder_scatter",info.context,para_s);
    }
    else scatter_kernel = KernelProcessor::getKernel("splitKernel.cl", "block_scatter", info.context, para_s);

    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * (buckets+1), NULL);

    if (reorder) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his_origin);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * local_buffer_len, NULL);
    }
    if (!key_only) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
        if(reorder)
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * local_buffer_len, NULL);
    }
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(info.currentQueue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    /*time report*/
    std::cout<<std::endl<<"Algo: WG-level\tData type: ";
    if (key_only)   std::cout<<"key-only\t";
    else            std::cout<<"key-value\t";
    std::cout<<"Reorder: ";
    if (reorder)   std::cout<<"yes"<<std::endl;
    else            std::cout<<"no"<<std::endl;

    std::cout<<"Total Time: "<<total_time<<" ms"<<std::endl;
    std::cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<std::endl;
    std::cout<<"\tScan Time: "<<scan_time<<" ms"<<std::endl;
    std::cout<<"\tScatter Time: " <<scatter_time<<" ms"<<std::endl;
    if (d_start != NULL)
        std::cout<<"\tGather time: "<<gather_time<<" ms."<<std::endl;

    clReleaseMemObject(d_his);
    if (reorder)    clReleaseMemObject(d_his_origin);
    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}

/*
 *  Thread-level partitioning (Each thread has a histogram)
 *  Input:  1.Table being partitioned,  (d_in_keys, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out_keys, d_out_values)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
*/
double thread_split(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, PlatInfo& info, cl_mem d_in_values, cl_mem d_out_values, int local_size, int grid_size) {

    //    localSize = 128, gridSize = 8192;
    bool key_only;
    if ((d_in_values != NULL) && (d_out_values != NULL)) key_only = false;
    else if ((d_in_values == NULL) && (d_out_values == NULL)) key_only = true;
    else {
        std::cerr <<"Wrong parameters."<< std::endl;
        return -1;
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    double totalTime = 0;

    cl_kernel histogram_kernel, gather_his_kernel, scatter_kernel;
    cl_mem d_his;
    double histogram_time, scan_time, gather_time, scatter_time, total_time=0;

    //kernel reading
    histogram_kernel = KernelProcessor::getKernel("splitKernel.cl", "thread_histogram", info.context);

    //set work group and NDRange sizes
    int global_size = local_size * grid_size;
    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)global_size};

/*1.histogram*/
    unsigned long his_len = buckets*global_size;

    //check whether the histogram can be placed in the global memory (at most 2^32 Bytes)
//    long his_len_comp = his_len;
//    long limit = 1<<32;
//    if (his_len_comp*sizeof(int) >= limit)   return 9999;

    d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogram_kernel, argsNum++, local_size*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(info.currentQueue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

/*2.scan*/
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len, info, 1024, 15, 0, 11);
    scan_time = scan_fast(d_his, his_len, info, 64, 39, 112, 0);
    total_time += scan_time;

/*2.5 gather the start position (optional)*/
    if (d_start != NULL) {
        gather_his_kernel = KernelProcessor::getKernel("splitKernel.cl", "gatherStartPos", info.context);
        argsNum = 0;
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &his_len);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_start);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &grid_size);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clEnqueueNDRangeKernel(info.currentQueue, gather_his_kernel, 1, 0, global, local, 0, 0, &event);
        status = clFinish(info.currentQueue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

/*3.scatter*/
    //compilation parameters
    char para_s[500] = {'\0'};
    if (!key_only)  strcat(para_s, "-DKVS ");
    scatter_kernel = KernelProcessor::getKernel("splitKernel.cl", "thread_scatter", info.context, para_s);

    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, local_size*buckets*sizeof(int), NULL);
    if(!key_only) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
    }
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    status = clEnqueueNDRangeKernel(info.currentQueue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    std::cout<<std::endl<<"Algo: WI-level\tData type: ";
    if (key_only)   std::cout<<"key-only"<<std::endl;
    else            std::cout<<"key-value"<<std::endl;
    std::cout<<"Total Time: "<<total_time<<" ms"<<std::endl;
    std::cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<std::endl;
    std::cout<<"\tScan Time: "<<scan_time<<" ms"<<std::endl;
    std::cout<<"\tScatter Time: " <<scatter_time<<" ms"<<std::endl;
    if (d_start != NULL)
        std::cout<<"\tGather time: "<<gather_time<<" ms."<<std::endl;

    clReleaseMemObject(d_his);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}

/*
 *  Single partitioning (local_size=1, for CPUs and MICs)
 *  Input:  1.Table being partitioned,  (d_in_keys, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out_keys, d_out_values)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
*/
double single_split(cl_mem d_in_keys, cl_mem d_out_keys, int length, int buckets, bool reorder, PlatInfo& info, cl_mem d_in_values, cl_mem d_out_values) {

    int local_size = 1, grid_size = 39;
    int len_per_group = (length + grid_size - 1)/grid_size;

    //check data type
    bool key_only;
    if ((d_in_values != NULL) && (d_out_values != NULL)) key_only = false;
    else if ((d_in_values == NULL) && (d_out_values == NULL)) key_only = true;
    else {
        std::cerr << "Wrong parameters." << std::endl;
        return -1;
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    cl_kernel histogram_kernel, scatter_kernel, gather_his_kernel;
    int *h_global_buffer_keys = NULL, *h_global_buffer_values = NULL;
    cl_mem d_his, d_global_buffer_keys, d_global_buffer_values;
    double histogram_time, scan_time, scatter_time, total_time = 0;

    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t) local_size};
    size_t global[1] = {(size_t) global_size};

/*1.histogram*/
    histogram_kernel = KernelProcessor::getKernel("splitKernel.cl", "single_histogram", info.context);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &len_per_group);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int)*buckets, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(info.currentQueue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

/*2.scan*/
//    double scanTime = scan_fast(d_his_in, d_his_out, his_len,  info, 1024, 15, 0, 11);
     scan_time = scan_fast(d_his, his_len, info, 64, 39, 112, 0);
//    scan_time = scan_fast(d_his, his_len, info, 64, 240, 33, 0);

    total_time += scan_time;

/*3.scatter*/
    //compilation parameters
    char para_s[500] = {'\0'};
    unsigned cacheline_size, ele_per_cacheline;
    if (!key_only)  strcat(para_s, " -DKVS ");
    if (reorder) {
        cacheline_size = 64;       /*in bytes*/
        ele_per_cacheline = cacheline_size/ sizeof(int);

        char para_str_1[20],para_str_2[20];
        my_itoa(cacheline_size, para_str_1, 10);
        strcat(para_s, " -DCACHELINE_SIZE=");
        strcat(para_s, para_str_1);

        my_itoa(ele_per_cacheline, para_str_2, 10);
        strcat(para_s, " -DELE_PER_CACHELINE=");
        strcat(para_s, para_str_2);

        scatter_kernel = KernelProcessor::getKernel("splitKernel.cl", "single_reorder_scatter",info.context,para_s);
    }
    else scatter_kernel = KernelProcessor::getKernel("splitKernel.cl", "single_scatter", info.context, para_s);

    /*alignment buffers*/
    h_global_buffer_keys = (int*)_mm_malloc(sizeof(int)*ele_per_cacheline*buckets*grid_size, cacheline_size);
    d_global_buffer_keys = clCreateBuffer(info.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(int)*ele_per_cacheline*buckets*grid_size, h_global_buffer_keys, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    if (!key_only) {
        h_global_buffer_values = (int*)_mm_malloc(sizeof(int)*ele_per_cacheline*buckets*grid_size, cacheline_size);
        d_global_buffer_values = clCreateBuffer(info.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(int)*ele_per_cacheline*buckets*grid_size, h_global_buffer_values, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
    }

    //end of test
    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_keys);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &len_per_group);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int)*buckets, NULL);

    if (reorder) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_global_buffer_keys);
    }
    if (!key_only) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
        if (reorder)
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_global_buffer_values);
    }
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(info.currentQueue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    std::cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<std::endl;

    /*time report*/
   // std::cout<<std::endl<<"Algo: Single\tData type: ";
   // if (key_only)   std::cout<<"key-only\t";
   // else            std::cout<<"key-value\t";
   // std::cout<<"Reorder: ";
   // if (reorder)   std::cout<<"yes"<<std::endl;
   // else            std::cout<<"no"<<std::endl;

   // std::cout<<"Total Time: "<<total_time<<" ms"<<std::endl;
   // std::cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<std::endl;
   // std::cout<<"\tScan Time: "<<scan_time<<" ms"<<std::endl;
   // std::cout<<"\tScatter Time: " <<scatter_time<<" ms"<<std::endl;

    clReleaseMemObject(d_his);
    clReleaseMemObject(d_global_buffer_keys);
    if(!key_only) clReleaseMemObject(d_global_buffer_values);

    if(h_global_buffer_keys) _mm_free(h_global_buffer_keys);
    if(h_global_buffer_values) _mm_free(h_global_buffer_values);

    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}
