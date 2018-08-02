//
//  splitImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/13/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Plat.h"
using namespace std;

/*
 *  WI-level partitioning (Each WI owns a private histogram)
 *  Input:  1.Table being partitioned,  (d_in, d_in_values)
 *          2.Table cadinality,         (length)
 *          3.Buckets                   (buckets)
 *  Output: 1.Partitioned table         (d_out, d_out_values)
 *          2.Array recording the start position of each bucket in the table (d_start)
 *
 *  If Data_structure is SOA, then d_in represents the input keys.
 *  If Data_structure is AOS, then d_in represents the input tuples, and the d_in_values, d_out_values shoule be set to 0
 *
*/
double WI_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets,
        Data_structure structure,
        cl_mem d_in_values, cl_mem d_out_values,
        int local_size, int grid_size)
{
    device_param_t param = Plat::get_device_param();

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        if ( (d_in_values == 0) || (d_out_values == 0) ) {
            cerr <<"Wrong parameters: values are not set."<< endl;
            return -1;
        }
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, NULL);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, NULL);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) )
        {
            cerr <<"Wrong parameters: inputs and outputs are not tuples"<< endl;
            return -1;
        }
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    cl_kernel histogram_kernel, gather_his_kernel, scatter_kernel;
    cl_mem d_his=0;
    double histogram_time, scan_time, gather_time, scatter_time, total_time=0;

    //set work group and NDRange sizes
    int global_size = local_size * grid_size;
    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)global_size};

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
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, his_len*sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(histogram_kernel, argsNum++, local_size*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
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

        status = clEnqueueNDRangeKernel(param.queue, gather_his_kernel, 1, 0, global, local, 0, 0, &event);
        status = clFinish(param.queue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

/*3.scatter*/
    scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WI_scatter", para_s);

    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out);
    if(structure == KVS_SOA) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
    }
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, local_size*buckets*sizeof(int), NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(splitWithListKernel);
#endif
    status = clEnqueueNDRangeKernel(param.queue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<endl;

    cout<<endl<<"Algo: WI-level\tData type: ";
    if (structure == KO)            cout<<"key-only"<<endl;
    else if (structure == KVS_AOS)  cout<<"key-value (AOS)"<<endl;
    else if (structure == KVS_SOA)  cout<<"key-value (SOA)"<<endl;

    cout<<"Local size: "<<local_size<<'\t'
             <<"Grid size: "<<grid_size<<endl;

    cout<<"Total Time: "<<total_time<<" ms"<<endl;
    cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<endl;
    cout<<"\tScan Time: "<<scan_time<<" ms"<<endl;
    cout<<"\tScatter Time: " <<scatter_time<<" ms"<<endl;
    if (d_start != NULL)
        cout<<"\tGather time: "<<gather_time<<" ms."<<endl;

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
 *  If Data_structure is SOA, then d_in represents the input keys.
 *  If Data_structure is AOS, then d_in represents the input tuples, and the d_in_values, d_out_values shoule be set to 0
 *
 *  Reorder:
 *      reorder = NO_REORDER: no reorder;
 *      reorder = FIXED_REORDER: with fixed-length reorder buffers  (lsize must be 1)
 *      reorder = VARIED_REORDER: with varied-length reorder buffers
 *
*/
double WG_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets, ReorderType reorder,
        Data_structure structure,
        cl_mem d_in_values, cl_mem d_out_values,
        int local_size, int grid_size)
{
    device_param_t param = Plat::get_device_param();
    uint64_t cus = param.cus;

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        if ( (d_in_values == 0) || (d_out_values == 0) ) {
            cerr <<"Wrong parameters: values are not set."<< endl;
            return -1;
        }
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, NULL);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, NULL);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) )
        {
            cerr <<"Wrong parameters: inputs and outputs are not tuples"<< endl;
            return -1;
        }
    }

    cl_int status = 0;
    cl_event event;
    int argsNum = 0;

    /*set the compilation paramters. Each kernel in the kernel file should be compilted with this parameter*/
    char para_s[500] = {'\0'};
    if (structure == KO)            strcat(para_s, " -DKO ");
    else if (structure == KVS_SOA)  strcat(para_s, " -DKVS_SOA ");
    else if (structure == KVS_AOS)  strcat(para_s, " -DKVS_AOS ");

//    checkLocalMemOverflow(sizeof(int) * buckets);    //this small, because of using atomic add

    cl_kernel histogram_kernel, scatter_kernel, gather_his_kernel;
    cl_mem d_his, d_his_origin;
    double histogram_time, gather_time, scan_time, scatter_time, total_time = 0;

    //set work group and NDRange sizes
    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;
    size_t local[1] = {(size_t) local_size};
    size_t global[1] = {(size_t) global_size};

/*1.histogram*/
    histogram_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_histogram", para_s);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int) * buckets, NULL);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

    //copy the global histogram before scan
    if (reorder) {
        d_his_origin = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * his_len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        status = clEnqueueCopyBuffer(param.queue, d_his, d_his_origin, 0, 0, sizeof(int) * his_len, 0, 0, 0);
        checkErr(status, ERR_EXEC_KERNEL);
        status = clFinish(param.queue);
    }

/*2.scan*/
//    double scanTime = scan_chained(d_his_in, d_his_out, his_len,  info, 1024, 15, 0, 11);
    scan_time = scan_chained(d_his, d_his, his_len, 64, cus-1, 112, 0);
//    scan_time = scan_chained(d_his, his_len, 64, 240, 33, 0);

    total_time += scan_time;

/*2.5 gather the start position (optional)*/
    if (d_start != NULL) {
        gather_his_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "gatherStartPos", para_s);
        argsNum = 0;
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &his_len);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(cl_mem), &d_start);
        status |= clSetKernelArg(gather_his_kernel, argsNum++, sizeof(int), &grid_size);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clEnqueueNDRangeKernel(param.queue, gather_his_kernel, 1, 0, global, local, 0, 0, &event);
        status = clFinish(param.queue);
        gather_time = clEventTime(event);
        total_time += gather_time;
    }

/*3.scatter*/
    if (reorder == FIXED_REORDER)
        scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle_fixed", para_s);
    else if (reorder == VARIED_REORDER)
        scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle_varied", para_s);
    else scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle", para_s);

    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out);
    if (structure == KVS_SOA) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
    }
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * (buckets+1), NULL);

    if (reorder) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his_origin);
        if (structure == KVS_AOS) { /*buffer for tuples */
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(tuple_t) * local_buffer_len, NULL);
        }
        else {          /*buffer for keys */
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem) * local_buffer_len, NULL);
            if (structure == KVS_SOA)   /*buffer for values */
                status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * local_buffer_len, NULL);
        }
    }
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clEnqueueNDRangeKernel(param.queue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<endl;

    /*time report*/
//    cout<<endl<<"Algo: WG-level\tData type: ";
//    if (structure == KO)            cout<<"key-only\t";
//    else if (structure == KVS_AOS)  cout<<"key-value (AOS)\t";
//    else if (structure == KVS_SOA)  cout<<"key-value (SOA)\t";
//    cout<<"Reorder: ";
//    if (reorder)   cout<<"yes"<<endl;
//    else            cout<<"no"<<endl;
//
//    cout<<"Local size: "<<local_size<<'\t'
//             <<"Grid size: "<<grid_size<<endl;
//
    cout<<"Total Time: "<<total_time<<" ms"<<endl;
    cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<endl;
    cout<<"\tScan Time: "<<scan_time<<" ms"<<endl;
    cout<<"\tScatter Time: " <<scatter_time<<" ms"<<endl;
    if (d_start != NULL)
        cout<<"\tGather time: "<<gather_time<<" ms."<<endl;

    clReleaseMemObject(d_his);
    if (reorder)    clReleaseMemObject(d_his_origin);
    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}

double single_split(
        cl_mem d_in, cl_mem d_out,
        int length, int buckets, bool reorder,
        cl_mem d_in_values, cl_mem d_out_values,
        Data_structure structure)
{
    device_param_t param = Plat::get_device_param();
    uint64_t cus = param.cus;

    int local_size = 1, grid_size = cus-1;
    int len_per_group = (length + grid_size - 1)/grid_size;

    /*check the value setting*/
    if (structure == KVS_SOA) { /*SOA should have both keys and values*/
        cerr <<"Wrong parameters: SOA not supported."<< endl;
        return -1;
    }
    /*check the key setting*/
    if (structure == KVS_AOS) {
        size_t in_mem_size, out_mem_size;
        clGetMemObjectInfo(d_in, CL_MEM_SIZE, sizeof(size_t), &in_mem_size, NULL);
        clGetMemObjectInfo(d_out, CL_MEM_SIZE, sizeof(size_t), &out_mem_size, NULL);
        if ( ( in_mem_size != length * sizeof(tuple_t) )||
             ( out_mem_size != length * sizeof(tuple_t)) )
        {
            cerr <<"Wrong parameters: inputs and outputs are not tuples"<< endl;
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
    int *h_global_buffer_int = NULL;
    tuple_t *h_global_buffer_tuple = NULL;
    double histogram_time, scan_time, scatter_time, total_time = 0;

    int global_size = local_size * grid_size;
    int local_buffer_len = length / grid_size;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t) local_size};
    size_t global[1] = {(size_t) global_size};

/*1.histogram*/
    histogram_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_histogram", para_s);
    int his_len = buckets * grid_size;
    d_his = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*his_len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int)*buckets, NULL);
    status |= clSetKernelArg(histogram_kernel, argsNum++, sizeof(int), &buckets);
    checkErr(status, ERR_SET_ARGUMENTS, -3);

    status = clEnqueueNDRangeKernel(param.queue, histogram_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    histogram_time = clEventTime(event);
    total_time += histogram_time;

/*2.scan*/
//    double scanTime = scan_chained(d_his_in, d_his_out, his_len,  info, 1024, 15, 0, 11);
     scan_time = scan_chained(d_his, d_his, his_len, 64, cus-1, 112, 0);
//    scan_time = scan_chained(d_his, his_len, info, 64, 240, 33, 0);

    total_time += scan_time;

/*3.scatter*/
    //compilation parameters

    if (reorder)
        scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle_fixed", para_s);
    else scatter_kernel = get_kernel(param.device, param.context, "split_kernel.cl", "WG_shuffle", para_s);

    /*alignment buffers*/
    int cacheline_size = 64;
    int ele_per_cacheline;
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

    argsNum = 0;
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out);
    if (structure == KVS_SOA) {
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_in_values);
        status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_out_values);
    }
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int), &buckets);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * (buckets+1), NULL);

    if (reorder) {
        if (structure == KVS_AOS) { /*buffer for tuples */
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(tuple_t) * local_buffer_len, NULL);
        }
        else {          /*buffer for keys */
            status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(cl_mem), &d_global_buffer);
            if (structure == KVS_SOA)   /*buffer for values */
                status |= clSetKernelArg(scatter_kernel, argsNum++, sizeof(int) * local_buffer_len, NULL);
        }
    }
    checkErr(status, ERR_SET_ARGUMENTS, -2);

    status = clEnqueueNDRangeKernel(param.queue, scatter_kernel, 1, 0, global, local, 0, 0, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    scatter_time = clEventTime(event);
    total_time += scatter_time;

    //*2 for keys and values, another *2 for read and write data
//    cout<<"Scatter time: "<<scatterTime<<" ms."<<"("<<length*sizeof(int)*2*2/scatterTime*1000/1e9<<"GB/s)"<<endl;

    /*time report*/
//    cout<<endl<<"Algo: Single\tData type: ";
//    if (structure == KO)            cout<<"key-only\t";
//    else if (structure == KVS_AOS)  cout<<"key-value (AOS)\t";
//    cout<<"Reorder: ";
//    if (reorder)   cout<<"yes"<<endl;
//    else            cout<<"no"<<endl;
//
//    cout<<"Local size: "<<local_size<<'\t'
//             <<"Grid size: "<<grid_size<<endl;
    cout<<"Total Time: "<<total_time<<" ms"<<endl;
    cout<<"\tHistogram Time: "<<histogram_time<<" ms"<<endl;
    cout<<"\tScan Time: "<<scan_time<<" ms"<<endl;
    cout<<"\tScatter Time: " <<scatter_time<<" ms"<<endl;

    clReleaseMemObject(d_his);
    clReleaseMemObject(d_global_buffer);

    if(h_global_buffer_int) _mm_free(h_global_buffer_int);
    if(h_global_buffer_tuple) _mm_free(h_global_buffer_tuple);

    checkErr(status, ERR_EXEC_KERNEL);

    return total_time;
}
