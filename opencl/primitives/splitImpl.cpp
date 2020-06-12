//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "../util/Plat.h"
#include "log.h"
using namespace std;

/*
 *  WI-level partitioning (Each WI owns a private histogram)
 *  Input:  1.Table being partitioned,  (d_in, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out, d_out_values)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
 *  If DataStruc is SOA, then d_in represents the input keys.
 *  If DataStruc is AOS, then d_in represents the input tuples, and the d_in_values, d_out_values should be set to 0
 *
*/
double WI_split(cl_mem d_in, cl_mem d_out, cl_mem d_start,
                int length, int buckets,
                DataStruc structure,
                cl_mem d_in_values, cl_mem d_out_values,
                int local_size, int grid_size) {
    device_param_t param = Plat::get_device_param();

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        if ( (d_in_values == 0) || (d_out_values == 0) ) {
            log_error("Wrong parameters: values are not set");
            return -1;
        }
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, nullptr);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, nullptr);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) ) {
            log_error("Wrong parameters: inputs and outputs are not tuples");
            return -1;
        }
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    cl_kernel histogram_kernel, gather_his_kernel, shuffle_kernel;
    cl_mem d_his=0;
    double histogram_time, scan_time, gather_time, shuffle_time, total_time=0;

    //set work group and NDRange sizes
    int global_size = local_size * grid_size;
    size_t local_dim[1] = {(size_t)local_size};
    size_t global_dim[1] = {(size_t)global_size};

    /*set the compilation paramters. Each kernel in the kernel file should be compilted with this parameter*/
    char para_s[500] = {'\0'};
    if (structure == KO)            strcat(para_s, " -DKO ");
    else if (structure == KVS_SOA)  strcat(para_s, " -DKVS_SOA ");
    else if (structure == KVS_AOS)  strcat(para_s, " -DKVS_AOS");

    /*1.histogram*/
    //kernel reading
    histogram_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WI_histogram", para_s);

    //check whether the histogram can be placed in the global memory (at most 2^32 Bytes)
//    long limit = 1<<32;
//    if (his_len_comp*sizeof(int) >= limit)   return 9999;

    /*hostogram allocation*/
    unsigned long his_len = buckets*global_size;
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, his_len*sizeof(int), nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogram_kernel, argsNum++, local_size*buckets*sizeof(int), nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);

    histogram_time = clEventTime(event);
    total_time += histogram_time;

    /*2.scan*/
//    double scanTime = scan_chained(d_his_in, d_his_out, his_len, info, 1024, 15, 0, 11);
    scan_time = scan_chained(d_his, d_his, his_len, 64, 39, 112, 0);
    total_time += scan_time;

    /*2.5 gather the start position (optional)*/
    if (d_start != 0) {
        gather_his_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "gatherStartPos", para_s);
        argsNum = 0;
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &his_len);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_start);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &grid_size);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clEnqueueNDRangeKernel(param.queue, gather_his_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

    /*3.shuffle*/
    shuffle_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WI_shuffle", para_s);

    argsNum = 0;
    status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(cl_mem), &d_out);
    if(structure == KVS_SOA) {
        status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
    }
    status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(shuffle_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(shuffle_kernel, argsNum++, local_size*buckets*sizeof(int), nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, shuffle_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    shuffle_time = clEventTime(event);
    total_time += shuffle_time;

    clReleaseMemObject(d_his);
    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}

/*
 *  WG-level partitioning (WIs in a WG share a histogram)
 *  Input:  1.Table being partitioned,  (d_in, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out, d_out_values)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
 *  If DataStruc is SOA, then d_in represents the input keys.
 *  If DataStruc is AOS, then d_in represents the input tuples, and the d_in_values, d_out_values shoule be set to 0
 *
 *  Reorder:
 *      reorder = NO_REORDER: no reorder;
 *      reorder = FIXED_REORDER: with fixed-length reorder buffers  (lsize must be 1)
 *      reorder = VARIED_REORDER: with varied-length reorder buffers
 *
*/
double WG_split(cl_mem d_in, cl_mem d_out, cl_mem d_start,
                int length, int buckets, ReorderType reorder_type,
                DataStruc structure,
                cl_mem d_in_values, cl_mem d_out_values,
                int local_size, int grid_size) {
    device_param_t param = Plat::get_device_param();
    uint64_t cus = param.cus;

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        if ( (d_in_values == 0) || (d_out_values == 0) ) {
            log_error("Wrong parameters: values are not set");
            return -1;
        }
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, nullptr);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, nullptr);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) ) {
            log_error("Wrong parameters: inputs and outputs are not tuples");
            return -1;
        }
    }

    if (reorder_type == FIXED_REORDER) {    /*fixed-length buffers only support single work-item*/
        local_size = 1;
        grid_size = cus;
    }

    cl_int status = 0;
    cl_event event;
    int args_num = 0;

    /*set the compilation paramters. Each kernel in the kernel file should be compilted with this parameter*/
    char para_s[500] = {'\0'};
    if (structure == KO)            strcat(para_s, " -DKO ");
    else if (structure == KVS_SOA)  strcat(para_s, " -DKVS_SOA ");
    else if (structure == KVS_AOS)  strcat(para_s, " -DKVS_AOS ");

//    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    cl_kernel histogram_kernel, shuffle_kernel, gather_his_kernel;
    int *h_global_buffer_int = nullptr, *h_global_buffer_int_values=nullptr;
    tuple_t *h_global_buffer_tuple = nullptr;
    cl_mem d_his=0, d_his_origin=0, d_global_buffer=0, d_global_buffer_values=0;
    double histogram_time, gather_time, scan_time, shuffle_time, total_time = 0;

    /*for fixed-length reorder buffers*/
    int cacheline_size = param.cacheline_size;
    int ele_per_cacheline;

    //set work group and NDRange sizes
    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;
    size_t local_dim[1] = {(size_t) local_size};
    size_t global_dim[1] = {(size_t) global_size};

    /*1.histogram*/
    histogram_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_histogram", para_s);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    args_num = 0;
    status |= clSetKernelArg(histogram_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, args_num++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, args_num++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, args_num++, sizeof(int) * buckets, nullptr);
    status |= clSetKernelArg(histogram_kernel, args_num++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

    //copy the global histogram before scan
    if (reorder_type == VARIED_REORDER) {
        d_his_origin = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        status = clEnqueueCopyBuffer(param.queue, d_his, d_his_origin, 0, 0, sizeof(int) * his_len, 0, 0, 0);
        checkErr(status, ERR_EXEC_KERNEL);
        status = clFinish(param.queue);
    }

    /*2.scan*/
//      scan_time = scan_chained(d_his, d_his, his_len, 1024, cus, 0, 11);
    scan_time = scan_chained(d_his, d_his, his_len, 64, cus-1, 112, 0);
//    scan_time = scan_chained(d_his, d_his, his_len, 64, 240, 33, 0);

    total_time += scan_time;

    /*2.5 gather the start position (optional)*/
    if (d_start != nullptr) {
        gather_his_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "gatherStartPos", para_s);
        args_num = 0;
        status |= clSetKernelArg(gather_his_kernel, args_num++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(gather_his_kernel, args_num++, sizeof(int), &his_len);
        status |= clSetKernelArg(gather_his_kernel, args_num++, sizeof(cl_mem), &d_start);
        status |= clSetKernelArg(gather_his_kernel, args_num++, sizeof(int), &grid_size);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clEnqueueNDRangeKernel(param.queue, gather_his_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

    /*3.shuffle*/
    if (reorder_type == FIXED_REORDER) {
        strcat(para_s, "-DCACHELINE_SIZE=");
        char cacheline_size_str[20];
        my_itoa(cacheline_size, cacheline_size_str, 10);
        strcat(para_s, cacheline_size_str);
        shuffle_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle_fixed", para_s);
    }
    else if (reorder_type == VARIED_REORDER)
        shuffle_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle_varied", para_s);
    else shuffle_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle", para_s);

    args_num = 0;
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_out);
    if (structure == KVS_SOA) {
        status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_out_values);
    }
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(int), &length);
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(int), &buckets);
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(int) * (buckets+1), nullptr);

    if (reorder_type == VARIED_REORDER) {           /*varied-length reorder buffers*/
        status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_his_origin);
        if (structure == KVS_AOS)
            status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(tuple_t) * local_buffer_len, nullptr);
        else {
            status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem) * local_buffer_len, nullptr);
            if (structure == KVS_SOA)   /*buffer for values */
                status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(int) * local_buffer_len, nullptr);
        }
    }
    else if (reorder_type == FIXED_REORDER) {       /*fixed-length reorder buffers*/
        /*alignment buffers*/
        if (structure == KVS_AOS) {
            ele_per_cacheline = cacheline_size / sizeof(tuple_t);
            h_global_buffer_tuple = (tuple_t*)_mm_malloc(sizeof(tuple_t)*ele_per_cacheline*buckets*grid_size, cacheline_size);
            d_global_buffer = clCreateBuffer(param.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(tuple_t)*ele_per_cacheline*buckets*grid_size, h_global_buffer_tuple, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_global_buffer);
        }
        else {
            ele_per_cacheline = cacheline_size / sizeof(int);
            h_global_buffer_int = (int*)_mm_malloc(sizeof(int)*ele_per_cacheline*buckets*grid_size, cacheline_size);
            d_global_buffer = clCreateBuffer(param.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(int)*ele_per_cacheline*buckets*grid_size, h_global_buffer_int, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_global_buffer);

            if (structure == KVS_SOA) {     /*buffer for values */
                h_global_buffer_int_values = (int *) _mm_malloc(sizeof(int) * ele_per_cacheline * buckets * grid_size, cacheline_size);
                d_global_buffer_values = clCreateBuffer(param.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(int)*ele_per_cacheline*buckets*grid_size, h_global_buffer_int_values, &status);
                checkErr(status, ERR_HOST_ALLOCATION);

                status |= clSetKernelArg(shuffle_kernel, args_num++, sizeof(cl_mem), &d_global_buffer_values);
            }
        }
    }
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, shuffle_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    shuffle_time = clEventTime(event);
    total_time += shuffle_time;

    /*memory release*/
    cl_mem_free(d_his_origin);
    cl_mem_free(d_his);
    cl_mem_free(d_global_buffer);
    cl_mem_free(d_global_buffer_values);

    if (h_global_buffer_int)            _mm_free(h_global_buffer_int);
    if (h_global_buffer_int_values)     _mm_free(h_global_buffer_int_values);
    if (h_global_buffer_tuple)          _mm_free(h_global_buffer_tuple);

    return total_time;
}

/*
 *  Single partitioning (local_size=1, for CPUs and MICs, only support AOS)
 *  Input:  1.Table being partitioned,  (d_in)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
 *
*/
double single_split(cl_mem d_in, cl_mem d_out,
                    int length, int buckets, bool reorder,
                    DataStruc structure) {
    device_param_t param = Plat::get_device_param();

    int local_size = 1, grid_size = 39;
    int len_per_group = (length + grid_size - 1)/grid_size;
    int cacheline_size, ele_per_cacheline;

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        log_error("Wrong parameters: SOA not supported");
        return -1;
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, nullptr);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, nullptr);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) ) {
            log_error("Wrong parameters: inputs and outputs are not tuples");
            return -1;
        }
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    /*set the compilation paramters. Each kernel in the kernel file should be compilted with this parameter*/
    char para_s[500] = {'\0'};
    if (structure == KO)            strcat(para_s, " -DKO ");
    else if (structure == KVS_AOS)  strcat(para_s, " -DKVS_AOS ");

    cl_kernel histogram_kernel, scatter_kernel, gather_his_kernel;
    cl_mem d_his, d_global_buffer, d_global_buffer_values;
    int *h_global_buffer_int = nullptr;
    tuple_t *h_global_buffer_tuple = nullptr;
    double histogram_time, scan_time, scatter_time, total_time = 0;

    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;

    //set work group and NDRange sizes
    size_t local_dim[1] = {(size_t) local_size};
    size_t global_dim[1] = {(size_t) global_size};

    /*1.histogram*/
    histogram_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "single_histogram", para_s);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*his_len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &len_per_group);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int)*buckets, nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

    /*2.scan*/
//    double scanTime = scan_chained(d_his_in, d_his_out, his_len,  info, 1024, 15, 0, 11);
    scan_time = scan_chained(d_his, d_his, his_len, 64, 39, 112, 0);
//    scan_time = scan_chained(d_his, d_his, his_len, info, 64, 240, 33, 0);

    total_time += scan_time;

    /*3.scatter*/
    if (reorder) {
        cacheline_size = param.cacheline_size;
        strcat(para_s, "-DCACHELINE_SIZE=");
        char cacheline_size_str[20];
        my_itoa(cacheline_size, cacheline_size_str, 10);
        strcat(para_s, cacheline_size_str);
        scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "single_fixed_shuffle", para_s);
    }
    else scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "single_shuffle", para_s);

    /*alignment buffers*/
    if (structure == KVS_AOS) {
        ele_per_cacheline = cacheline_size / sizeof(tuple_t);
        h_global_buffer_tuple = (tuple_t*)_mm_malloc(sizeof(tuple_t)*ele_per_cacheline*buckets*grid_size, cacheline_size);
        d_global_buffer = clCreateBuffer(param.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(tuple_t)*ele_per_cacheline*buckets*grid_size, h_global_buffer_tuple, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
    }
    else {
        ele_per_cacheline = cacheline_size / sizeof(int);
        h_global_buffer_int = (int*)_mm_malloc(sizeof(int)*ele_per_cacheline*buckets*grid_size, cacheline_size);
        d_global_buffer = clCreateBuffer(param.context, CL_MEM_USE_HOST_PTR| CL_MEM_READ_WRITE, sizeof(int)*ele_per_cacheline*buckets*grid_size, h_global_buffer_int, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
    }

    //end of test
    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &len_per_group);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int)*buckets, nullptr);

    if (reorder) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_global_buffer);
    }
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, scatter_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    clReleaseMemObject(d_his);
    clReleaseMemObject(d_global_buffer);

    if(h_global_buffer_int) _mm_free(h_global_buffer_int);
    if(h_global_buffer_tuple) _mm_free(h_global_buffer_tuple);

    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}