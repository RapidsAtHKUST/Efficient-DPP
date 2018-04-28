//
//  phjImpl.cpp: non-partition based hash join
//  gpuqp_opencl
//
//  Created by Bryan on 5/16/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double build_np(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_table_keys, cl_mem d_table_values, unsigned hash_bits, PlatInfo info);
double probe_np(cl_mem d_S_keys, cl_mem d_S_values, int s_len, cl_mem d_table_keys, cl_mem d_table_values, int hash_bits, PlatInfo info, int &res_len);

//testing: only count the number of outputs
double hashjoin_np(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len, PlatInfo info)
{
    struct timeval start, end;
    double totalTime = 0;
    cl_int status;

    unsigned hash_bits = 1;
    while ((1<<hash_bits) < (2*rLen)) {
        hash_bits++;
    }
    unsigned table_len = 1<<hash_bits;
    std::cout<<"Hash table len:"<<table_len<<std::endl;

    cl_mem d_table_keys = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*table_len,NULL,&status);
    cl_mem d_table_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*table_len,NULL,&status);

    std::cout<<"Begin to build hash table."<<std::endl;
    build_np(d_R_keys, d_R_values, rLen, d_table_keys, d_table_values, hash_bits, info);
    std::cout<<"Hash table built."<<std::endl;
    probe_np(d_S_keys, d_S_values, sLen, d_table_keys, d_table_values, hash_bits, info, res_len);
    std::cout<<"Probe finished."<<std::endl;

    return totalTime;
}

//build the shared hash table
double build_np(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_table_keys, cl_mem d_table_values, unsigned hash_bits, PlatInfo info)
{
    cl_int status;
    int localSize = 1024;
    int gridSize = 1024;
    int globalSize = localSize * gridSize;

    cl_kernel buildKernel = KernelProcessor::getKernel("hjNonPartitionedKernel.cl", "build", info.context);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(globalSize)};

    int argsNum = 0;
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(cl_mem), &d_R_keys);
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(cl_mem), &d_R_values);
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(cl_mem), &d_table_keys);
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(cl_mem), &d_table_values);
    status |= clSetKernelArg(buildKernel, argsNum++, sizeof(unsigned), &hash_bits);

    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(buildKernel);
#endif

    struct timeval start, end;
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, buildKernel, 1, 0, global, local, 0, 0, 0 );
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);

    double totalTime = diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);

    return totalTime;
}

double probe_np(cl_mem d_S_keys, cl_mem d_S_values, int s_len, cl_mem d_table_keys, cl_mem d_table_values, int hash_bits, PlatInfo info, int &res_len)
{
    cl_int status;
    int argsNum = 0;
    double totalTime = 0;
    int gridSize = 1024;              //Each work-group works for a bucket-bucket pair
    int localSize = 1024;
    int globalSize = localSize * gridSize;

    //kernel reading
    cl_kernel probeKernel = KernelProcessor::getKernel("hjNonPartitionedKernel.cl", "probe", info.context);

    //memory allocation
//    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*globalSize, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);

    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(globalSize)};

    struct timeval start, end;

    //testing: only counting the number of output elements
    cl_mem d_out_num = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    int h_out_num = 0;
    status = clEnqueueWriteBuffer(info.currentQueue, d_out_num, CL_TRUE, 0, sizeof(int), &h_out_num, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
//
    argsNum = 0;
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_S_keys);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_S_values);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &s_len);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_table_keys);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_table_values);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &hash_bits);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_out_num);

    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(matchCountKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, probeKernel, 1, 0, global, mylocal, 0, 0, 0 );
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);

    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);

    status = clEnqueueReadBuffer(info.currentQueue, d_out_num, CL_TRUE, 0, sizeof(int), &h_out_num, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    res_len = h_out_num;
    std::cout<<"# Joined result:"<<h_out_num<<std::endl;

    return totalTime;
}
