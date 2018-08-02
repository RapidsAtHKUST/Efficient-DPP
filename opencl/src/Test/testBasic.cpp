//
//  TestBasic.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 6/14/16.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Plat.h"
using namespace std;

/*
 * Test scalar multiplication bandwidth
 */
void testMem() {
    device_param_t param = Plat::get_device_param();

    cl_int status = 0;
    cl_event event;
    int argsNum;
    int localSize = 1024;
    int gridSize = 262144;
    double mulTime = 0.0;
    int scalar = 13;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    //get the kernel
    cl_kernel mul_kernel = get_kernel(param.device, param.context, "memKernel.cl", "mul_bandwidth");

    int len = localSize*gridSize;
    std::cout<<"Data size for read/write(multiplication test): "<<len<<" ("<<len*sizeof(int)*1.0/1024/1024<<"MB)"<<std::endl;

    //data initialization
    int *h_in = new int[len];
    for(int i = 0; i < len; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_WRITE , sizeof(int)*len, NULL, &status);
    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY , sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(int), &scalar);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(param.queue);
    for(int i = 0; i < EXPERIMENT_TIMES; i++) {
        status = clEnqueueNDRangeKernel(param.queue, mul_kernel, 1, 0, global, local, 0, 0, &event);  //kernel execution
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);

        //throw away the first result
        if (i != 0) mulTime += clEventTime(event);
    }
    mulTime /= (EXPERIMENT_TIMES - 1);

    status = clFinish(param.queue);
    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_in;

    //compute the bandwidth, including read and write
    double throughput = computeMem(len*2, sizeof(int), mulTime);
    cout<<"Time for multiplication: "<<mulTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput<<" GB/s"<<endl;
}

void testAccess() {
    device_param_t param = Plat::get_device_param();

    cl_event event;
    cl_int status = 0;
    int argsNum = 0;
    int localSize = 1024, gridSize = 8192;
    int repeat_max = 100;
    int length = localSize * gridSize * repeat_max;
    std::cout<<"Maximal data size: "<<length<<" ("<<length* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    //kernel reading
    cl_kernel mul_row_kernel = get_kernel(param.device, param.context, "memKernel.cl", "mul_row_based");
    cl_kernel mul_column_kernel = get_kernel(param.device, param.context, "memKernel.cl", "mul_column_based");
    cl_kernel mul_mixed_kernel = get_kernel(param.device, param.context, "memKernel.cl", "mul_mixed");

    //memory allocation
    int *h_in = new int[length];
    int *h_out = new int[length];
    for(int i = 0; i < length; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int)*length, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clFinish(param.queue);

    //set kernel arguments: mul_coalesced_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_column_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_column_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_strided_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_row_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_row_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_warped_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_mixed_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_mixed_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    clFlush(param.queue);
    status = clFinish(param.queue);

    //time recorder
    double *column_time = new double[repeat_max+1];
    double *column_throughput = new double[repeat_max+1];
    double *row_time = new double[repeat_max+1];
    double *row_throughput = new double[repeat_max+1];
    double *mixed_time = new double[repeat_max+1];
    double *mixed_throughput = new double[repeat_max+1];

    for(int i = 1; i <= repeat_max; i++) {
        column_time[i] = 0.0;
        column_throughput[i] = 0.0;
        row_time[i] = 0.0;
        row_throughput[i] = 0.0;
        mixed_time[i] = 0.0;
        mixed_throughput[i] = 0.0;
    }

    //coalesced
    cout<<"------------------ Coalesced Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(mul_column_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, mul_column_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     column_time[re] += tempTime;
        }
    }

    for(int re = 1; re <= repeat_max; re++) {
        column_time[re] /= (EXPERIMENT_TIMES - 1);
        assert(column_time[re] > 0);
        column_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), column_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<column_time[re]<<" ms\t"<<"Throughput: "<<column_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    //strided
    cout<<"------------------ Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(mul_row_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, mul_row_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     row_time[re] += tempTime;
        }
    }

    //warpwise strided
    for(int re = 1; re <= repeat_max; re++) {
        row_time[re] /= (EXPERIMENT_TIMES - 1);
        assert(row_time[re] > 0);
        row_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), row_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<row_time[re]<<" ms\t"<<"Throughput: "<<row_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    cout<<"------------------ Warpwise Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < EXPERIMENT_TIMES; i++) {
            status |= clSetKernelArg(mul_mixed_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(param.queue, mul_mixed_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(param.queue);
            status = clFinish(param.queue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     mixed_time[re] += tempTime;
        }
    }

    for(int re = 1; re <= repeat_max; re++) {
        mixed_time[re] /= (EXPERIMENT_TIMES - 1);
        assert(mixed_time[re] > 0);
        mixed_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), mixed_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<mixed_time[re]<<" ms\t"<<"Throughput: "<<mixed_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    status = clFinish(param.queue);

    //for python formating
    cout<<endl;
    cout<<"strided_bandwidth = ["<<row_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<row_throughput[re];
    }
    cout<<"]"<<endl;

    cout<<"coalesced_bandwidth = ["<<column_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<column_throughput[re];
    }
    cout<<"]"<<endl;

    cout<<"mix_bandwidth = ["<<mixed_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<mixed_throughput[re];
    }
    cout<<"]"<<endl;

    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);

    if(h_in) delete [] h_in;
    if(h_out) delete [] h_out;

    if(column_time) delete[] column_time;
    if(column_throughput) delete[] column_throughput;
    if(row_time) delete[] row_time;
    if(row_throughput) delete[] row_throughput;
    if(mixed_time) delete[] mixed_time;
    if(mixed_throughput) delete[] mixed_throughput;
}


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
    return averageHampel(tempTimes, EXPERIMENT_TIMES);
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

       throughput = computeMem(len, sizeof(int), aveTime);
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
