//
//  phjImpl.cpp: partition based hash join
//  gpuqp_opencl
//
//  Created by Bryan on 5/16/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Plat.h"

double probe(cl_mem d_R_keys, cl_mem d_R_values, cl_mem d_S_keys, cl_mem d_S_values, int r_len, int s_len, cl_mem start_R, cl_mem start_S, int buckets, int &res_len);

//testing: only count the number of outputs
double hashjoin(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len)
{
    device_param_t param = Plat::get_device_param();
    Data_structure structure = KVS_SOA; /*SOA by default*/

    struct timeval start, end;
    double totalTime = 0;
    cl_int status;
    int bits = 13;          //grid size = 1024
    int buckets = (1<<bits);

    cl_mem d_R_partitioned_keys = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_R_partitioned_values = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_S_partitioned_keys = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_S_partitioned_values = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem r_start = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*buckets, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem s_start = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*buckets, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

//    gettimeofday(&start, NULL);
    double r_time = WG_split(d_R_keys, d_R_partitioned_keys, r_start, rLen, buckets, false, structure, d_R_values, d_R_partitioned_values);
//    gettimeofday(&end, NULL);
//    double r_time = diffTime(end, start);

//    gettimeofday(&start, NULL);
    double s_time = WG_split(d_S_keys, d_S_partitioned_keys , s_start, sLen, buckets, false, structure, d_S_values, d_S_partitioned_values);
//    gettimeofday(&end, NULL);
//    double s_time = diffTime(end, start);

//    gettimeofday(&start, NULL);
    double probeTime = probe(d_R_partitioned_keys, d_R_partitioned_values, d_S_partitioned_keys, d_S_partitioned_values, rLen, sLen, r_start, s_start, buckets, res_len);
//    gettimeofday(&end, NULL);
//    double probeTime = diffTime(end, start);

    totalTime += r_time;
    totalTime += s_time;
    totalTime += probeTime;

    std::cout<<"R partition time: "<<r_time<<" ms."<<std::endl;
    std::cout<<"S partition time: "<<s_time<<" ms."<<std::endl;
    std::cout<<"Probe time: "<<probeTime<<" ms."<<std::endl;

    return totalTime;
}

double probe(cl_mem d_R_keys, cl_mem d_R_values, cl_mem d_S_keys, cl_mem d_S_values, int r_len, int s_len, cl_mem start_R, cl_mem start_S, int buckets, int &res_len)
{
    cl_int status;
    int argsNum = 0;
    double totalTime = 0;
    int gridSize = buckets;              //Each work-group works for a bucket-bucket pair
    int localSize = 1024;
    int globalSize = localSize * gridSize;

    //kernel reading
    device_param_t param = Plat::get_device_param();
    cl_kernel probeKernel = get_kernel(param.device, param.context, "hj_partitioned_kernel.cl", "build_probe");

    //memory allocation
//    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*globalSize, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);

    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(globalSize)};

    struct timeval start, end;

    //testing: only counting the number of output elements
    cl_mem d_out_num = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    int h_out_num = 0;
    status = clEnqueueWriteBuffer(param.queue, d_out_num, CL_TRUE, 0, sizeof(int), &h_out_num, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
//
    argsNum = 0;
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_R_keys);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_R_values);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &r_len);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_S_keys);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_S_values);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &s_len);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &start_R);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &start_S);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_out_num);
    status |= clSetKernelArg(probeKernel, argsNum++, 8*1024, NULL);    //15.5KB for R partition  (keys)
     status |= clSetKernelArg(probeKernel, argsNum++, 8*1024, NULL);    //15.5KB for R partition (values)
    status |= clSetKernelArg(probeKernel, argsNum++, 15.5*1024, NULL);    //15.5KB for S partition hash table (keys)
    status |= clSetKernelArg(probeKernel, argsNum++, 15.5*1024, NULL);    //15.5KB for S partition hash table (values)

    checkErr(status, ERR_SET_ARGUMENTS);

    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(param.queue, probeKernel, 1, 0, global, mylocal, 0, 0, 0 );
    status = clFinish(param.queue);
    gettimeofday(&end, NULL);

    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);

    status = clEnqueueReadBuffer(param.queue, d_out_num, CL_TRUE, 0, sizeof(int), &h_out_num, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    res_len = h_out_num;
//    std::cout<<"# Joined result:"<<h_out_num<<std::endl;

    return totalTime;
}
