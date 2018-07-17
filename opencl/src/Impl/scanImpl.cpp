//
//  scanImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Plat.h"
using namespace std;

/*
 *  grid size should be equal to the # of computing units
 *  R: number of elements in registers in each work-item
 *  L: number of elememts in local memory
 */
double scan_fast(cl_mem &d_inout, int length, int local_size, int grid_size, int R, int L)
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

    cl_kernel scanBlockKernel = Plat::get_kernel("scanKernel.cl", "scan_fast", extra);

    //initialize the intermediate array
    int *h_inter = new int[num_of_blocks];
    for(int i = 0; i < num_of_blocks; i++) h_inter[i] = -1;

    cl_mem d_inter = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)* num_of_blocks, NULL, &status);
    status = clEnqueueWriteBuffer(param.queue, d_inter, CL_TRUE, 0, sizeof(int)*num_of_blocks, h_inter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size * grid_size)};

    //help, for debug
//    cl_mem d_help = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*local_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int)*local_mem_size, NULL);    //local memory lo
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &lo_size);           //local mem size
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &num_of_blocks);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &R);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &L);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_inter);

//    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_help);

    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(scanBlockKernel);
#endif
    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scanBlockKernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

//    int *h_help = new int[local_size];
//    status = clEnqueueReadBuffer(param.queue, d_help, CL_TRUE, 0, sizeof(int)*local_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < local_size; i++) cout<<i<<' '<<h_help[i]<<endl;
//    cout<<endl;
//    delete[] h_help;
//    clReleaseMemObject(d_help);

    clReleaseMemObject(d_inter);
    if(h_inter) delete[] h_inter;

    return totalTime;
}

double scan_three_kernel(cl_mem &d_inout, unsigned length, int local_size, int grid_size)
{
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;
    int global_size = local_size * grid_size;

    const int reduce_ele_per_wi = (length + global_size -1)/global_size;
    const int scan_ele_per_wi = 14;      //temporary number

    //conpilation parameters
    char extra[500], para_reduce[20], para_scan[20];
    my_itoa(reduce_ele_per_wi, para_reduce, 10);       //transfer R to string
    my_itoa(scan_ele_per_wi, para_scan, 10);       //transfer R to string

    strcpy(extra, "-DREDUCE_ELE_PER_WI=");
    strcat(extra, para_reduce);
    strcat(extra, " -DSCAN_ELE_PER_WI=");
    strcat(extra, para_scan);
//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = Plat::get_kernel("scan_rss_kernel.cl", "reduce", extra);

    size_t reduce_local[1] = {(size_t)local_size};
    size_t reduce_global[1] = {(size_t)(global_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    //help, for debug
//    cl_mem d_help = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int)*grid_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, reduce_kernel, 1, 0, reduce_global, reduce_local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

//-------------------------- Step 2: intermediate scan -----------------------------
    cl_kernel scan_small_kernel = Plat::get_kernel("scan_rss_kernel.cl", "scan_exclusive_small", extra); //still need extra paras

    size_t scan_small_local[1] = {(size_t)local_size};
    size_t scan_small_global[1] = {(size_t)(local_size*1)};

    argsNum = 0;
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int), &grid_size);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int)*grid_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_small_kernel, 1, 0, scan_small_global, scan_small_local, 0, NULL, &event); //single WG execution
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

//-------------------------- Step 3: final exclusive scan -----------------------------
    cl_kernel scan_kernel = Plat::get_kernel("scan_rss_kernel.cl", "scan_exclusive", extra);

    int scan_len_per_wg = (length + grid_size - 1)/grid_size;

    size_t scan_local[1] = {(size_t)local_size};
    size_t scan_global[1] = {(size_t)(local_size*grid_size)};

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &scan_len_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int)*local_size*scan_ele_per_wi, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, scan_global, scan_local, 0, NULL, &event);
    status = clFinish(param.queue);

    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, scan_global, scan_local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

//    std::cout<<"local_size:"<<local_size<<'\t'<<"grid_size:"<<grid_size<<'\t'<<"ele_per_wg:"<<length/grid_size<<std::endl;
//    int *h_help = new int[grid_size];
//    status = clEnqueueReadBuffer(param.queue, d_reduction, CL_TRUE, 0, sizeof(int)*grid_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < grid_size; i++) {
//        cout<<h_help[i]<<' ';
//    }
//    cout<<endl;
//    for(int i = 1; i < grid_size; i++) {
//        if (h_help[i] - h_help[i-1]!= 1024) {
//            cout<<i<<' '<<h_help[i]<<' '<<h_help[i-1]<<endl;
//        }
//    }
//    cout<<endl;
//    delete[] h_help;
    clReleaseMemObject(d_reduction);

    cout<<endl;
    cout<<"reduce time:"<<reduce_time<<" ms"<<endl;
    cout<<"scan small time:"<<scan_small_time<<" ms"<<endl;
    cout<<"scan exclusive time:"<<scan_time<<" ms"<<endl;

    return totalTime;
}

//single-threaded
double scan_three_kernel_single(cl_mem &d_inout, unsigned length, int grid_size)
{
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    //single-threaded
    int local_size = 1;
    int len_per_wg = (length + grid_size - 1) / grid_size;

//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = Plat::get_kernel("scan_rss_single_kernel.cl", "reduce_single");

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size*grid_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_inout);
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
    cl_kernel scan_small_kernel = Plat::get_kernel("scan_rss_single_kernel.cl", "scan_no_offset_single"); //still need extra paras

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

    cl_kernel scan_kernel = Plat::get_kernel("scan_rss_single_kernel.cl", "scan_with_offset_single");

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

//    std::cout<<"local_size:"<<local_size<<'\t'<<"grid_size:"<<grid_size<<'\t'<<"ele_per_wg:"<<length/grid_size<<std::endl;
//    int *h_help = new int[grid_size];
//    status = clEnqueueReadBuffer(param.queue, d_reduction, CL_TRUE, 0, sizeof(int)*grid_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < grid_size; i++) {
//        cout<<h_help[i]<<' ';
//    }
//    cout<<endl;
//    delete[] h_help;

    clReleaseMemObject(d_reduction);
    return totalTime;
}

