//
//  TestBasic.cpp
//  gpuqp_opencl
//
//  Created by Zhuohang Lai on 6/14/16.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#include "Plat.h"
using namespace std;

double wg_sequence(int *h_data, int len, int *h_idx_array, int local_size, int grid_size, bool loaded) {
    device_param_t param = Plat::get_device_param();

    cl_event event;
    cl_int status;
    int h_atom = 0;
    int args_num;

    cl_kernel kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "wg_access");

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(local_size)};
    size_t global[1] = {(size_t)(local_size * grid_size)};
    int len_per_wg = len / grid_size;

    double tempTimes[EXPERIMENT_TIMES];
    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*len, NULL, &status);
        cl_mem d_idx_arr = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*grid_size, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(param.queue, d_idx_arr, CL_TRUE, 0, sizeof(int)*grid_size, h_idx_array, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);

        cl_mem d_atom = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int), NULL, &status);
        status = clEnqueueFillBuffer(param.queue, d_atom, &h_atom, sizeof(int), 0, sizeof(int), 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);

        //flush the cache
        if(loaded) {
            cl_kernel heat_kernel = get_kernel(param.device, param.context, "mem_kernel.cl", "cache_heat");
            args_num = 0;
            status |= clSetKernelArg(heat_kernel, args_num++, sizeof(cl_mem), &d_in);
            status |= clSetKernelArg(heat_kernel, args_num++, sizeof(int), &len);
            checkErr(status, ERR_SET_ARGUMENTS);

            status = clEnqueueNDRangeKernel(param.queue, heat_kernel, 1, 0, global, local, 0, 0, 0);
            status = clFinish(param.queue);
            checkErr(status, ERR_EXEC_KERNEL,1);
        }
        args_num = 0;
        status |= clSetKernelArg(kernel, args_num++, sizeof(cl_mem), &d_in);
        status |= clSetKernelArg(kernel, args_num++, sizeof(int), &len_per_wg);
        status |= clSetKernelArg(kernel, args_num++, sizeof(cl_mem), &d_atom);
        status |= clSetKernelArg(kernel, args_num++, sizeof(cl_mem), &d_idx_arr);
        checkErr(status, ERR_SET_ARGUMENTS);

        //kernel execution
        status = clFinish(param.queue);

        //kernel execution
        struct timeval start, end;

        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(param.queue, kernel, 1, 0, global, local, 0, 0, &event);
        status = clFinish(param.queue);
        gettimeofday(&end, NULL);

        checkErr(status, ERR_EXEC_KERNEL,2);

        // status = clEnqueueReadBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int)*len, h_data, 0, 0, 0);
        // for(int i = 0; i < 1048576; i++) {
        //     cout<<h_data[i]<<' ';
        // }
        // cout<<endl;

        //free the memory
        status = clReleaseMemObject(d_in);
        status = clReleaseMemObject(d_idx_arr);
        status = clReleaseMemObject(d_atom);
        checkErr(status, ERR_RELEASE_MEM);

        // tempTimes[e] = clEventTime(event);
        tempTimes[e] = diffTime(end, start);
    }
    return average_Hampel(tempTimes, EXPERIMENT_TIMES);
}

void test_wg_sequence(unsigned long len) {
    device_param_t param = Plat::get_device_param();

    std::cout<<"Data size : "<<len<<" ("<<len*sizeof(int)*1.0/1024/1024<<"MB)"<<std::endl;

    int local_size = 1;
    double aveTime, throughput;

    //size of the partition each work-group processes
    int ds_wg_kb, ds_wg_kb_begin = 8, ds_wg_kb_end=8192;

    int *h_data = new int[len];
    for(int i = 0; i < len; i++) h_data[i] = 1;

//----------------------- case 1: Interleaved, unloaded -------------------------------
   cout<<"Case: Interleaved, unloaded."<<endl;
   for(ds_wg_kb = ds_wg_kb_end; ds_wg_kb >= ds_wg_kb_begin; ds_wg_kb >>= 1) {
       int grid_size = len/1024/ ds_wg_kb * sizeof(int);
       cout<<"ele_per_wi:"<<len/grid_size/local_size<<endl;
       int *h_idx_array = new int[grid_size];
       for(int i = 0; i < grid_size; i++) h_idx_array[i] = i;

       aveTime = wg_sequence(h_data, len, h_idx_array, local_size, grid_size, false);
        // aveTime = wg_sequence_no_atomic(h_data,len,local_size,grid_size,info,false);

       throughput = compute_bandwidth(len, sizeof(int), aveTime);
       cout<<"\tDS/WG:"<<ds_wg_kb<<"KB "<<"gs:"<<grid_size<<" Time: "<<aveTime<<" ms."<<'\t'
           <<"Throughput: "<<throughput<<" GB/s"<<endl;
       if (h_idx_array)    delete[] h_idx_array;
   }

//----------------------- case 2: Interleaved, loaded -------------------------------
    // cout<<"Case: Interleaved, loaded."<<endl;
    // for(ds_wg_kb = ds_wg_kb_begin; ds_wg_kb <= ds_wg_kb_end; ds_wg_kb <<= 1) {
    //     int grid_size = len/1024/ ds_wg_kb * sizeof(int);
    //     int *h_idx_array = new int[grid_size];
    //     for(int i = 0; i < grid_size; i++) h_idx_array[i] = i;

    //     aveTime = wg_sequence(h_data, len, h_idx_array, local_size, grid_size, info, true);
    //     throughput = computeMem(len, sizeof(int), aveTime);
    //     cout<<"\tDS/WG:"<<ds_wg_kb<<"KB "<<"gs:"<<grid_size<<" Time: "<<aveTime<<" ms."<<'\t'
    //         <<"Throughput: "<<throughput<<" GB/s"<<endl;

    //     if (h_idx_array)    delete[] h_idx_array;
    // }

//----------------------- case 3: Random, unloaded -------------------------------

//    cout<<"Case: Random, unloaded."<<endl;
//    for(ds_wg_kb = ds_wg_kb_begin; ds_wg_kb <= ds_wg_kb_end; ds_wg_kb <<= 1) {
//        int grid_size = len/1024/ ds_wg_kb * sizeof(int);
//        int *h_idx_array = new int[grid_size];
//        for(int i = 0; i < grid_size; i++) h_idx_array[i] = i;
//        //shuffle the idx_array
//        srand(time(NULL));
//        for(int i = 0; i <grid_size * 3; i++) {
//            int idx1 = rand() % grid_size;
//            int idx2 = rand() % grid_size;
//            int temp = h_idx_array[idx1];
//            h_idx_array[idx1] = h_idx_array[idx2];
//            h_idx_array[idx2] = temp;
//        }
//
//        aveTime = wg_sequence(h_data, len, h_idx_array,local_size, grid_size, info, false);
//        throughput = computeMem(len, sizeof(int), aveTime);
//        cout<<"\tDS/WG:"<<ds_wg_kb<<"KB "<<"gs:"<<grid_size<<" Time: "<<aveTime<<" ms."<<'\t'
//            <<"Throughput: "<<throughput<<" GB/s"<<endl;
//
//        if (h_idx_array)    delete[] h_idx_array;
//    }

//----------------------- case 4: Random, loaded -------------------------------
//    cout<<"Case: Random, loaded."<<endl;
//    for(ds_wg_kb = ds_wg_kb_begin; ds_wg_kb <= ds_wg_kb_end; ds_wg_kb <<= 1) {
//        int grid_size = len/1024/ ds_wg_kb * sizeof(int);
//        int *h_idx_array = new int[grid_size];
//        for(int i = 0; i < grid_size; i++) h_idx_array[i] = i;
//
//        //shuffle the idx_array
//        srand(time(NULL));
//        for(int i = 0; i <grid_size * 3; i++) {
//            int idx1 = rand() % grid_size;
//            int idx2 = rand() % grid_size;
//            int temp = h_idx_array[idx1];
//            h_idx_array[idx1] = h_idx_array[idx2];
//            h_idx_array[idx2] = temp;
//        }
//
//        aveTime = wg_sequence(h_data, len, h_idx_array,local_size, grid_size, info, true);
//        throughput = computeMem(len, sizeof(int), aveTime);
//        cout<<"\tDS/WG:"<<ds_wg_kb<<"KB "<<"gs:"<<grid_size<<" Time: "<<aveTime<<" ms."<<'\t'
//            <<"Throughput: "<<throughput<<" GB/s"<<endl;
//
//        if (h_idx_array)    delete[] h_idx_array;
//    }

// wg_row, not neccessary
//    int num_wgs_per_cu = (grid_size + num_CUs - 1) / num_CUs;
//    for(int i = 0; i < num_wgs_per_cu; i++) {
//        for(int j = 0; j < num_CUs; j++) {
//            if (i*num_CUs+j >= grid_size)   break;
//            h_idx_array[i*num_CUs+j] = j*num_wgs_per_cu+i;
//        }
//    }
//    aveTime = wg_sequence(h_data, len, h_idx_array,local_size, grid_size, info, false);
//    throughput = computeMem(len*2, sizeof(int), aveTime);
//    cout<<"Case: wg_row(in+out)\tTime: "<<aveTime<<" ms."<<'\t'
//        <<"Throughput: "<<throughput<<" GB/s"<<endl;
//
//    aveTime = wg_sequence(h_data, len, h_idx_array,local_size, grid_size, info, true);
//    throughput = computeMem(len*2, sizeof(int), aveTime);
//    cout<<"Case: wg_row(in only)\tTime: "<<aveTime<<" ms."<<'\t'
//        <<"Throughput: "<<throughput<<" GB/s"<<endl;

    if(h_data) delete[] h_data;
}
