//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "Plat.h"
#include "log.h"

bool test_bandwidth() {
    log_trace("Function: %s", __FUNCTION__);
    device_param_t param = Plat::get_device_param();

    cl_event event;
    int args_num;
    cl_int status = 0;
    int local_size = 1024, grid_size = 262144, scalar = 13;
    double copy_time = 0.0, scale_time = 0.0, addition_time = 0.0, triad_time = 0.0;
    double copy_throughput, scale_throughput, addition_throughput, triad_throughput;

    size_t local_dim[1] = {(size_t)(local_size)};
    size_t global_dim[1] = {(size_t)(local_size * grid_size)};

    /* get the kernels */
    auto copy_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "copy_bandwidth");
    auto scale_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "scale_bandwidth");
    auto addition_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "addition_bandwidth");
    auto triad_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "triad_bandwidth");

    auto len = local_size*grid_size;
    log_info("Data cardinality: %d (%.1f MB)", len, 1.0*len*sizeof(int)/1024/1024);

    //data initialization
    int *h_in_1 = new int[len];
    int *h_in_2 = new int[len];
    for(int i = 0; i < len; i++) {
        h_in_1[i] = i;
        h_in_2[i] = i + 10;
    }
    cl_mem d_in_1 = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*len, nullptr, &status);
    cl_mem d_in_2 = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*len, nullptr, &status);
    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY , sizeof(int)*len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_in_1, CL_TRUE, 0, sizeof(int)*len, h_in_1, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(param.queue, d_in_2, CL_TRUE, 0, sizeof(int)*len, h_in_2, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    /* --- copy kernel --- */
    args_num = 0;
    status |= clSetKernelArg(copy_kernel, args_num++, sizeof(cl_mem), &d_in_1);
    status |= clSetKernelArg(copy_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        status = clEnqueueNDRangeKernel(param.queue, copy_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);
        if (i != 0) copy_time += clEventTime(event); //throw away the first result
    }
    copy_time /= (EXPERIMENT_TIMES - 1);
    status = clFinish(param.queue);

    /* --- scale kernel --- */
    args_num = 0;
    status |= clSetKernelArg(scale_kernel, args_num++, sizeof(cl_mem), &d_in_1);
    status |= clSetKernelArg(scale_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        status = clEnqueueNDRangeKernel(param.queue, scale_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);
        if (i != 0) scale_time += clEventTime(event); //throw away the first result
    }
    scale_time /= (EXPERIMENT_TIMES - 1);
    status = clFinish(param.queue);

    /* --- addition kernel --- */
    args_num = 0;
    status |= clSetKernelArg(addition_kernel, args_num++, sizeof(cl_mem), &d_in_1);
    status |= clSetKernelArg(addition_kernel, args_num++, sizeof(cl_mem), &d_in_2);
    status |= clSetKernelArg(addition_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        status = clEnqueueNDRangeKernel(param.queue, addition_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);
        if (i != 0) addition_time += clEventTime(event); //throw away the first result
    }
    addition_time /= (EXPERIMENT_TIMES - 1);
    status = clFinish(param.queue);

    /* --- triad kernel --- */
    args_num = 0;
    status |= clSetKernelArg(triad_kernel, args_num++, sizeof(cl_mem), &d_in_1);
    status |= clSetKernelArg(triad_kernel, args_num++, sizeof(cl_mem), &d_in_2);
    status |= clSetKernelArg(triad_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        status = clEnqueueNDRangeKernel(param.queue, triad_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);
        if (i != 0) triad_time += clEventTime(event); //throw away the first result
    }
    triad_time /= (EXPERIMENT_TIMES - 1);
    status = clFinish(param.queue);

    status = clReleaseMemObject(d_in_1);
    status = clReleaseMemObject(d_in_2);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_in_1;
    delete[] h_in_2;

    /* compute the bandwidth */
    log_info("Copy: time=%.1f ms, bandwidth=%.1f GB/s",
             copy_time, compute_bandwidth(len*2, sizeof(int), copy_time));
    log_info("Scale: time=%.1f ms, bandwidth=%.1f GB/s",
             scale_time, compute_bandwidth(len*2, sizeof(int), scale_time));
    log_info("Addition: time=%.1f ms, bandwidth=%.1f GB/s",
             addition_time, compute_bandwidth(len*3, sizeof(int), addition_time));
    log_info("Triad: time=%.1f ms, bandwidth=%.1f GB/s",
             triad_time, compute_bandwidth(len*3, sizeof(int), triad_time));

    return true;
}

/*
 * Usage:
 *    ./test_bandwidth
 * */
int main(int argc, const char *argv[]) {
    Plat::plat_init();
    assert(test_bandwidth());
    return 0;
}