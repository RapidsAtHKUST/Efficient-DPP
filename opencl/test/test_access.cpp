//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "Plat.h"
#include "log.h"

bool test_access() {
    log_trace("Function: %s", __FUNCTION__);
    device_param_t param = Plat::get_device_param();

    cl_event event;
    cl_int status = 0;
    int args_num = 0;
    int localSize = 512, gridSize = 32768, repeat_max = 20;
    int length = localSize * gridSize * repeat_max;
    log_info("Maximal data size: %d (%.1f MB)", length, 1.0*length* sizeof(int)/1024/1024);
    log_info("WARP_SIZE=%d, WARP_BITS=%d", WARP_SIZE, WARP_BITS);

    /* get the kernels */
    cl_kernel scale_row_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "scale_row");
    cl_kernel scale_column_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "scale_column");
    cl_kernel scale_mixed_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "scale_mixed");

    //memory allocation
    int *h_in = new int[length];
    int *h_out = new int[length];
    for(int i = 0; i < length; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_ONLY , sizeof(int)*length, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY , sizeof(int)*length, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int)*length, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clFinish(param.queue);

    //set kernel arguments: mul_coalesced_kernel
    args_num = 0;
    status |= clSetKernelArg(scale_column_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scale_column_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_strided_kernel
    args_num = 0;
    status |= clSetKernelArg(scale_row_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scale_row_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_warped_kernel
    args_num = 0;
    status |= clSetKernelArg(scale_mixed_kernel, args_num++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scale_mixed_kernel, args_num++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local_dim[1] = {(size_t)localSize};
    size_t global_dim[1] = {(size_t)(localSize * gridSize)};

    clFlush(param.queue);
    status = clFinish(param.queue);

    log_info("------------------ Column-major access ------------------");
    for(int re = 1; re <= repeat_max; re++) {
        double kernel_time = 0.0;
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(scale_column_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, scale_column_kernel, 1, 0,
                                            global_dim, local_dim, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);
            if (i != 0)     kernel_time += tempTime;
        }
        kernel_time /= (EXPERIMENT_TIMES - 1);
        log_info("Data size=%d*%d*%d (%.1f MB), time=%.1f ms, throughput=%.1f GB/s",
                 localSize, gridSize, re, 1.0*localSize*gridSize*re* sizeof(int)/1024/1024, kernel_time,
                 compute_bandwidth((uint64_t)localSize*gridSize*(re)*2, sizeof(int), kernel_time));
    }

    log_info("------------------ Row-major access ------------------");
    for(int re = 1; re <= repeat_max; re++) {
        double kernel_time = 0.0;
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(scale_row_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, scale_row_kernel, 1, 0,
                                            global_dim, local_dim, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);
            if (i != 0)     kernel_time += tempTime;
        }
        kernel_time /= (EXPERIMENT_TIMES - 1);
        log_info("Data size=%d*%d*%d (%.1f MB), time=%.1f ms, throughput=%.1f GB/s",
                 localSize, gridSize, re, 1.0*localSize*gridSize*re* sizeof(int)/1024/1024, kernel_time,
                 compute_bandwidth((uint64_t)localSize*gridSize*(re)*2, sizeof(int), kernel_time));
    }

    log_info("------------------ Mixed access ------------------");
    for(int re = 1; re <= repeat_max; re++) {
        double kernel_time = 0.0;
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(scale_mixed_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, scale_mixed_kernel, 1, 0,
                                            global_dim, local_dim, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);
            if (i != 0)     kernel_time += tempTime;
        }
        kernel_time /= (EXPERIMENT_TIMES - 1);
        log_info("Data size=%d*%d*%d (%.1f MB), time=%.1f ms, throughput=%.1f GB/s",
                 localSize, gridSize, re, 1.0*localSize*gridSize*re* sizeof(int)/1024/1024, kernel_time,
                 compute_bandwidth((uint64_t)localSize*gridSize*(re)*2, sizeof(int), kernel_time));
    }
    status = clFinish(param.queue);

    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);

    if(h_in) delete [] h_in;
    if(h_out) delete [] h_out;

    return true;
}

/*
 * Usage:
 *    ./test_access
 * */
int main(int argc, const char *argv[]) {
    Plat::plat_init();
    assert(test_access());
    return 0;
}