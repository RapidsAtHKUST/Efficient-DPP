//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "Plat.h"
#include "log.h"

double access_wg(int *h_data, int len, int *h_idxes, int local_size, int grid_size, bool loaded) {
    device_param_t param = Plat::get_device_param();
    cl_event event;
    cl_int status;
    int args_num, h_atom = 0;
    cl_kernel wg_access_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "wg_access");

    //set work group and NDRange sizes
    size_t local_dim[1] = {(size_t)(local_size)};
    size_t global_dim[1] = {(size_t)(local_size * grid_size)};
    int len_per_wg = len / grid_size;

    double tempTimes[EXPERIMENT_TIMES];
    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*len, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        cl_mem d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int), nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        cl_mem d_atom = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int), nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        status = clEnqueueFillBuffer(param.queue, d_atom, &h_atom, sizeof(int), 0, sizeof(int), 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
        cl_mem d_idxes = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*grid_size, nullptr, &status);
        checkErr(status, ERR_WRITE_BUFFER);

        status = clEnqueueWriteBuffer(param.queue, d_idxes, CL_TRUE, 0, sizeof(int)*grid_size, h_idxes, 0, 0, 0);

        //flush the cache
        if(loaded) {
            cl_kernel heat_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "cache_heat");
            args_num = 0;
            status |= clSetKernelArg(heat_kernel, args_num++, sizeof(cl_mem), &d_in);
            status |= clSetKernelArg(heat_kernel, args_num++, sizeof(int), &len);
            checkErr(status, ERR_SET_ARGUMENTS);

            status = clEnqueueNDRangeKernel(param.queue, heat_kernel, 1, 0, global_dim, local_dim, 0, 0, 0);
            status = clFinish(param.queue);
            checkErr(status, ERR_EXEC_KERNEL,1);
        }
        args_num = 0;
        status |= clSetKernelArg(wg_access_kernel, args_num++, sizeof(cl_mem), &d_in);
        status |= clSetKernelArg(wg_access_kernel, args_num++, sizeof(int), &len_per_wg);
        status |= clSetKernelArg(wg_access_kernel, args_num++, sizeof(cl_mem), &d_atom);
        status |= clSetKernelArg(wg_access_kernel, args_num++, sizeof(cl_mem), &d_idxes);
        status |= clSetKernelArg(wg_access_kernel, args_num++, sizeof(cl_mem), &d_out);
        checkErr(status, ERR_SET_ARGUMENTS);
        status = clFinish(param.queue);

        //kernel execution
        status = clEnqueueNDRangeKernel(param.queue, wg_access_kernel, 1, 0, global_dim, local_dim, 0, 0, &event);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL,2);

        //free the memory
        status = clReleaseMemObject(d_in);
        status = clReleaseMemObject(d_out);
        status = clReleaseMemObject(d_atom);
        checkErr(status, ERR_RELEASE_MEM);

        tempTimes[e] = clEventTime(event);
    }
    return average_Hampel(tempTimes, EXPERIMENT_TIMES);
}

bool test_access_wg(unsigned long len) {
    device_param_t param = Plat::get_device_param();
    log_info("Data size=%d (%.1f MB)", len, 1.0*len*sizeof(int)*1.0/1024/1024);

    //size of the partition each work-group processes
    int local_size = 1; //each warp activates a single thread
    int size_per_wg_kb, size_per_wg_kb_begin=8, size_per_wg_kb_end=8192;

    int *h_data = new int[len];
    for(int i = 0; i < len; i++) h_data[i] = 1;

    log_info("-------- Case: Interleaved, unloaded --------");
    for(size_per_wg_kb = size_per_wg_kb_end; size_per_wg_kb >= size_per_wg_kb_begin; size_per_wg_kb >>= 1) {
        auto grid_size = len/1024/ size_per_wg_kb * sizeof(int);
        int *h_idxes = new int[grid_size];
        for(auto i = 0; i < grid_size; i++) h_idxes[i] = i;
        auto ave_time = access_wg(h_data, len, h_idxes, local_size, grid_size, false);
        log_info("Data size/WG=%d KB, Items/WI=%d, grid_size=%d, time=%.1f ms, throughput=%.1f GB/s",
                 size_per_wg_kb, len/grid_size/local_size, grid_size, ave_time,
                 compute_bandwidth(len, sizeof(int), ave_time));
        delete[] h_idxes;
    }
    log_info("-------- Case: Interleaved, loaded --------");
    for(size_per_wg_kb = size_per_wg_kb_end; size_per_wg_kb >= size_per_wg_kb_begin; size_per_wg_kb >>= 1) {
        auto grid_size = len/1024/ size_per_wg_kb * sizeof(int);
        int *h_idxes = new int[grid_size];
        for(auto i = 0; i < grid_size; i++) h_idxes[i] = i;
        auto ave_time = access_wg(h_data, len, h_idxes, local_size, grid_size, true);
        log_info("Data size/WG=%d KB, Items/WI=%d, grid_size=%d, time=%.1f ms, throughput=%.1f GB/s",
                 size_per_wg_kb, len/grid_size/local_size, grid_size, ave_time,
                 compute_bandwidth(len, sizeof(int), ave_time));
        delete[] h_idxes;
    }
    log_info("-------- Case: Random, unloaded --------");
    for(size_per_wg_kb = size_per_wg_kb_end; size_per_wg_kb >= size_per_wg_kb_begin; size_per_wg_kb >>= 1) {
        auto grid_size = len/1024/ size_per_wg_kb * sizeof(int);
        int *h_idxes = new int[grid_size];
        for(int i = 0; i < grid_size; i++) h_idxes[i] = i;
        //shuffle the idx_array
        srand(time(nullptr));
        for(int i = grid_size-1; i > 0; i--) {
            int idx1 = rand() % i;
            int idx2 = rand() % i;
            std::swap(h_idxes[idx1], h_idxes[idx2]);
        }
        auto ave_time = access_wg(h_data, len, h_idxes, local_size, grid_size, false);
        log_info("Data size/WG=%d KB, Items/WI=%d, grid_size=%d, time=%.1f ms, throughput=%.1f GB/s",
                 size_per_wg_kb, len/grid_size/local_size, grid_size, ave_time,
                 compute_bandwidth(len, sizeof(int), ave_time));
        delete[] h_idxes;
    }
    log_info("-------- Case: Random, loaded --------");
    for(size_per_wg_kb = size_per_wg_kb_end; size_per_wg_kb >= size_per_wg_kb_begin; size_per_wg_kb >>= 1) {
        auto grid_size = len/1024/ size_per_wg_kb * sizeof(int);
        int *h_idxes = new int[grid_size];
        for(int i = 0; i < grid_size; i++) h_idxes[i] = i;
        //shuffle the idx_array
        srand(time(nullptr));
        for(int i = grid_size-1; i > 0; i--) {
            int idx1 = rand() % i;
            int idx2 = rand() % i;
            std::swap(h_idxes[idx1], h_idxes[idx2]);
        }
        auto ave_time = access_wg(h_data, len, h_idxes, local_size, grid_size, true);
        log_info("Data size/WG=%d KB, Items/WI=%d, grid_size=%d, time=%.1f ms, throughput=%.1f GB/s",
                 size_per_wg_kb, len/grid_size/local_size, grid_size, ave_time,
                 compute_bandwidth(len, sizeof(int), ave_time));
        delete[] h_idxes;
    }

    delete[] h_data;
    return true;
}

/*
 * Usage:
 *    ./test_access_wg DATA_SIZE
 * */
int main(int argc, const char *argv[]) {
    Plat::plat_init();
    auto data_size = stoull(argv[1]);
    assert(test_access_wg(data_size));
    return 0;
}