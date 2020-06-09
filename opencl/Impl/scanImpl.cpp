//
//  scanImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "../util/Plat.h"
using namespace std;

/*
 *  grid size should be equal to the # of computing units
 *  R: number of elements in registers in each work-item
 *  L: number of elememts in local memory
 */
double scan_chained(cl_mem d_in, cl_mem d_out, int length, int local_size, int grid_size, int R, int L)
{
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    int local_size_log = log2(local_size);
    int tile_size = local_size * (R + L);
    int num_of_blocks = (length + tile_size - 1) / tile_size;

    //kernel reading
    char extra[500];        
    strcpy(extra, "-DREGISTERS=");
    char R_li[20];

    //calculating the demanding intermediate local memory size
    int lo_size;            //intermediate memory size
    int local_mem_size;      //actual memory size

    if (R==0 && L==0) {
        cerr<<"Parameter error. R and L can not be 0 at the same time."<<endl;
        return 1;
    }

    if (R==0)   lo_size = L * local_size;
    else        lo_size = (L+1)*local_size;

    if (lo_size > local_size * R)       local_mem_size = lo_size;
    else                                local_mem_size = local_size * R;

    int DR;
    if (R == 0) DR = 1;
    else        DR = R;
    my_itoa(DR, R_li, 10);       //transfer R to string
    strcat(extra, R_li);

    cl_kernel kernel = get_kernel(param.device, param.context, "scan_kernel.cl", "scan", extra);

    //initialize the intermediate array
    int *h_inter = new int[num_of_blocks];
    for(int i = 0; i < num_of_blocks; i++) h_inter[i] = -1;

    cl_mem d_inter = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)* num_of_blocks, NULL, &status);
    status = clEnqueueWriteBuffer(param.queue, d_inter, CL_TRUE, 0, sizeof(int)*num_of_blocks, h_inter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size * grid_size)};

    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int)*local_mem_size, NULL);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &lo_size);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &num_of_blocks);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &R);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &L);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_inter);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    clReleaseMemObject(d_inter);
    if(h_inter) delete[] h_inter;

    return totalTime;
}

/*
 *  Single-pass single-threaded scan for CPUs and MICs
 *  tile_size: number of items in each tile (partition)
 */
double scan_global_single_thread(cl_mem d_in, cl_mem d_out, int length, int tile_size)
{
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    int num_of_partitions = (length + tile_size - 1) / tile_size;

    int local_size = 1;
    int grid_size = num_of_partitions;

    /*kernel reading*/
    cl_kernel kernel = get_kernel(param.device, param.context, "scan_kernel.cl", "scan_global_mem");

    /*initialize the intermediate array*/
    int *h_inter = new int[num_of_partitions];
    for(int i = 0; i < num_of_partitions; i++) h_inter[i] = -1;

    int par_counter = 0;

    cl_mem d_inter = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)* num_of_partitions, NULL, &status);
    status = clEnqueueWriteBuffer(param.queue, d_inter, CL_TRUE, 0, sizeof(int)*num_of_partitions, h_inter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    cl_mem d_par_counter = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);
    status = clEnqueueWriteBuffer(param.queue, d_par_counter, CL_TRUE, 0, sizeof(int), &par_counter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size * grid_size)};

    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &tile_size);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_par_counter);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_inter);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    clReleaseMemObject(d_inter);
    if(h_inter) delete[] h_inter;

    return totalTime;
}

/*multi-WI RSS scan for GPUs*/
double scan_rss(cl_mem d_in, cl_mem d_out, unsigned length, int local_size, int grid_size)
{
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;
    int global_size = local_size * grid_size;

    if (grid_size > length) grid_size = length; /*for cases with only a few tuples but lots of WGs*/
    const int reduce_ele_per_wg = (length + grid_size -1)/grid_size;            /*used in step 1*/
    const int scan_ele_per_wi = 5;     /*used in step 3, to ensure that 2 WGs executed in a CU*/
    const int scan_ele_per_wg = (length + grid_size - 1)/grid_size;
    int scan_ele_per_loop = (scan_ele_per_wi*local_size < scan_ele_per_wg) ? (scan_ele_per_wi*local_size) : scan_ele_per_wg; /*number of elements processed in each iteration in step 3*/
    const int max_reg_per_WI = 10;  /*for 1024 threads, each thread has at most 16 registers*/

    //conpilation parameters
    char param_str[1000] = {'\0'};
    add_param(param_str, "REDUCE_ELE_PER_WG", true, reduce_ele_per_wg);
    add_param(param_str, "SCAN_ELE_PER_LOOP", true, scan_ele_per_loop);
    add_param(param_str, "MAX_NUM_REGS", true, max_reg_per_WI);

//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = get_kernel(param.device, param.context, "scan_rss_kernel.cl", "reduce", param_str);

    size_t reduce_local[1] = {(size_t)local_size};
    size_t reduce_global[1] = {(size_t)(global_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    //help, for debug
//    cl_mem d_help = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int)*local_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, reduce_kernel, 1, 0, reduce_global, reduce_local, 0, NULL, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    clFinish(param.queue);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

//    int *h_reduction = new int[grid_size];
//    status = clEnqueueReadBuffer(param.queue, d_reduction, CL_TRUE, 0, sizeof(int)*grid_size, h_reduction, 0, 0, 0);
//    for(int i = 0; i < grid_size; i++) {
//        cout<<i<<' '<<h_reduction[i]<<' '<<endl;
//    }
//    cout<<endl;
//    delete[] h_reduction;
//    cout<<"reduce_ele_per_wg:"<<reduce_ele_per_wg<<endl;

//-------------------------- Step 2: intermediate scan -----------------------------
    cl_kernel scan_small_kernel = get_kernel(param.device, param.context, "scan_rss_kernel.cl", "scan_exclusive_small", param_str); //still need extra paras

    size_t scan_small_local[1] = {(size_t)local_size};
    size_t scan_small_global[1] = {(size_t)(local_size*1)};

    argsNum = 0;
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int), &grid_size);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int)*grid_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_small_kernel, 1, 0, scan_small_global, scan_small_local, 0, NULL, &event); //single WG execution
    checkErr(status, ERR_EXEC_KERNEL);
    clFinish(param.queue);

    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

//-------------------------- Step 3: final exclusive scan -----------------------------
    cl_kernel scan_kernel = get_kernel(param.device, param.context, "scan_rss_kernel.cl", "scan_exclusive", param_str);

    size_t scan_local[1] = {(size_t)local_size};
    size_t scan_global[1] = {(size_t)(local_size*grid_size)};

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &scan_ele_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int)*local_size*scan_ele_per_wi, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, scan_global, scan_local, 0, NULL, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    status = clFinish(param.queue);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

    clReleaseMemObject(d_reduction);

//    cout<<endl;
//    cout<<"reduce time:"<<reduce_time<<" ms"<<endl;
//    cout<<"scan small time:"<<scan_small_time<<" ms"<<endl;
//    cout<<"scan exclusive time:"<<scan_time<<" ms"<<endl;

    return totalTime;
}

/*single-WI RSS scan for CPUs and MICs*/
double scan_rss_single(cl_mem d_in, cl_mem d_out, unsigned length)
{
    device_param_t param = Plat::get_device_param();
    int grid_size = 1024;           /*this does not matter*/
    int local_size = 1;             /*single-work-item*/
    int len_per_wg = (length + grid_size - 1) / grid_size;

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = get_kernel(param.device, param.context, "scan_rss_single_kernel.cl", "reduce_single");

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size*grid_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, reduce_kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

//-------------------------- Step 2: intermediate scan -----------------------------
    cl_kernel scan_small_kernel = get_kernel(param.device, param.context, "scan_rss_single_kernel.cl", "scan_no_offset_single"); //still need extra paras

    size_t small_local[1] = {(size_t)(1)};
    size_t small_global[1] = {(size_t)(1)};

    argsNum = 0;
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int), &grid_size);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_small_kernel, 1, 0, small_global, small_local, 0, NULL, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

//-------------------------- Step 3: final exclusive scan  -----------------------------

    cl_kernel scan_kernel = get_kernel(param.device, param.context, "scan_rss_single_kernel.cl", "scan_with_offset_single");

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, global, local, 0, NULL, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);

    double scan_time = clEventTime(event);
    totalTime += scan_time;

    clReleaseMemObject(d_reduction);

//    cout<<endl;
//    cout<<"reduce time:"<<reduce_time<<" ms"<<endl;
//    cout<<"scan small time:"<<scan_small_time<<" ms"<<endl;
//    cout<<"scan exclusive time:"<<scan_time<<" ms"<<endl;

    return totalTime;
}

