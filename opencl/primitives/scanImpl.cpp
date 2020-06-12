//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#include "../util/Plat.h"
#include "log.h"
using namespace std;

/*
 *  grid size should be equal to the # of computing units
 *  R: number of elements in registers in each work-item
 *  L: number of elememts in local memory
 */
double
scan_chained(cl_mem d_in, cl_mem d_out,
             int length, int local_size,
             int grid_size, int R, int L) {
    if (R==0 && L==0) {
        log_error("Parameter error. R and L can not be 0 at the same time");
        return 1;
    }
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int args_num = 0;
    char extra_flags[500] = "\0"; //extra flages
    int val_invalid = SCAN_INTER_INVALID;
    int tile_size = local_size * (R + L);
    int num_tiles = (length + tile_size - 1) / tile_size;
    auto lo_size = (R == 0) ? L*local_size : (L+1)*local_size; //intermediate memory size
    auto local_mem_size = std::max(lo_size, local_size*R); //actual memory size

    sprintf(extra_flags, "-DREGISTERS=%d", (R==0) ? 1 : R); //specify the REGISTERS macro
    cl_kernel chain_scan_kernel = get_kernel(param.device, param.context, "scan_global_chain_kernel.cl", "scan", extra_flags);

    cl_mem d_inter = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)* num_tiles, nullptr, &status);
    status = clEnqueueFillBuffer(param.queue, d_inter, &val_invalid, sizeof(int), 0, sizeof(int)*num_tiles, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    size_t local_dim[1] = {(size_t)local_size};
    size_t global_dim[1] = {(size_t)(local_size * grid_size)};

    args_num = 0;
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(int), &length);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(int)*local_mem_size, nullptr);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(int), &num_tiles);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(int), &R);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(int), &L);
    status |= clSetKernelArg(chain_scan_kernel, args_num++, sizeof(cl_mem), &d_inter);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, chain_scan_kernel, 1, 0, global_dim, local_dim, 0, nullptr, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

    clReleaseMemObject(d_inter);

    return totalTime;
}

/* Ruduce-Scan-Scan scheme for GPUs*/
double scan_RSS(cl_mem d_in, cl_mem d_out, unsigned length, int local_size, int grid_size) {
    log_trace("Function: %s", __FUNCTION__);
    device_param_t param = Plat::get_device_param();

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int args_num = 0;
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

    /*--------------- Step 1: reduce ---------------*/
    cl_kernel reduce_kernel = get_kernel(param.device, param.context, "scan_global_RSS_kernel.cl", "reduce", param_str);

    size_t reduce_local[1] = {(size_t)local_size};
    size_t reduce_global[1] = {(size_t)(global_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, nullptr, &status);

    args_num = 0;
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(int), &length);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(int)*local_size, nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, reduce_kernel, 1, 0, reduce_global, reduce_local, 0, nullptr, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    clFinish(param.queue);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

    /*--------------- Step 2: scan ---------------*/
    cl_kernel scan_small_kernel = get_kernel(param.device, param.context, "scan_global_RSS_kernel.cl", "scan_exclusive_small", param_str); //still need extra paras

    size_t scan_small_local[1] = {(size_t)local_size};
    size_t scan_small_global[1] = {(size_t)(local_size*1)};

    args_num = 0;
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(int), &grid_size);
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(int)*grid_size, nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_small_kernel, 1, 0, scan_small_global, scan_small_local, 0, nullptr, &event); //single WG execution
    checkErr(status, ERR_EXEC_KERNEL);
    clFinish(param.queue);

    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

    /*--------------- Step 3: final scan ---------------*/
    cl_kernel scan_kernel = get_kernel(param.device, param.context, "scan_global_RSS_kernel.cl", "scan_exclusive", param_str);

    size_t scan_local[1] = {(size_t)local_size};
    size_t scan_global[1] = {(size_t)(local_size*grid_size)};

    args_num = 0;
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(int), &scan_ele_per_wg);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(int), &length);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(int)*local_size*scan_ele_per_wi, nullptr);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, scan_global, scan_local, 0, nullptr, &event);
    checkErr(status, ERR_EXEC_KERNEL);
    status = clFinish(param.queue);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

    clReleaseMemObject(d_reduction);

    return totalTime;
}

/*single-thread RSS scan for CPUs and MICs*/
double scan_RSS_single(cl_mem d_in, cl_mem d_out, unsigned length) {
    device_param_t param = Plat::get_device_param();
    int grid_size = 1024;           /*this does not matter*/
    int local_size = 1;             /*single-work-item*/
    int len_per_wg = (length + grid_size - 1) / grid_size;

    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int args_num = 0;

    /*--------------- Step 1: reduce ---------------*/
    cl_kernel reduce_kernel = get_kernel(param.device, param.context, "scan_global_RSS_single_kernel.cl", "reduce");

    size_t local_dim[1] = {(size_t)local_size};
    size_t global_dim[1] = {(size_t)(local_size*grid_size)};

    cl_mem d_reduction = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, nullptr, &status);

    args_num = 0;
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(reduce_kernel, args_num++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, reduce_kernel, 1, 0, global_dim, local_dim, 0, nullptr, &event);
    status = clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

    /*--------------- Step 2: scan ---------------*/
    cl_kernel scan_small_kernel = get_kernel(param.device, param.context, "scan_global_RSS_single_kernel.cl", "scan_small"); //still need extra paras

    size_t small_local[1] = {(size_t)(1)};
    size_t small_global[1] = {(size_t)(1)};

    args_num = 0;
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, args_num++, sizeof(int), &grid_size);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_small_kernel, 1, 0, small_global, small_local, 0, nullptr, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

    /*--------------- Step 3: final scan ---------------*/
    cl_kernel scan_kernel = get_kernel(param.device, param.context, "scan_global_RSS_single_kernel.cl", "scan_exclusive");

    args_num = 0;
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(scan_kernel, args_num++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    status = clEnqueueNDRangeKernel(param.queue, scan_kernel, 1, 0, global_dim, local_dim, 0, nullptr, &event);
    clFinish(param.queue);
    checkErr(status, ERR_EXEC_KERNEL);

    double scan_time = clEventTime(event);
    totalTime += scan_time;

    clReleaseMemObject(d_reduction);

    return totalTime;
}

