//
//  phjImpl.cpp: partition based hash join
//  gpuqp_opencl
//
//  Created by Bryan on 5/16/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double probe(cl_mem d_R, cl_mem d_S, int r_len, int s_len, cl_mem start_R, cl_mem start_S, int buckets, PlatInfo info, int &res_len);

/*  Radix partitioning
 *  totalCountBits: partition bits
 */
//double partitionHJ(cl_mem& d_R, int rLen,int totalCountBits, PlatInfo info, int localSize, int gridSize) {
//
//    double totalTime = 0;
//
//    cl_int status;
//    int itemNum = gridSize * localSize;
//
//    char splitPath[100] = PROJECT_ROOT;
//    strcat(splitPath, "/Kernels/splitKernel.cl");
//    std::string splitKerAddr = splitPath;
//
//    char hjPath[100] = PROJECT_ROOT;
//    strcat(hjPath, "/Kernels/hjKernel.cl");
//    std::string hjKerAddr = hjPath;
//
//    char createListHJSource[100] = "createListHJ";
//    char splitWithListHJSource[100] = "splitWithListHJ";
//
//    KernelProcessor splitReader(&splitKerAddr,1,info.context, "");
//    KernelProcessor hjReader(&hjKerAddr,1,info.context, "");
//
//    cl_kernel createListHJKernel = splitReader.getKernel(createListHJSource);
//    cl_kernel splitWithListHJKernel = splitReader.getKernel(splitWithListHJSource);
//
//    //set work group and NDRange sizes
//    size_t mylocal[1] = {(size_t)localSize};
//    size_t global[1] = {(size_t)(localSize * gridSize)};
//
//    int bits = 4;
//    int radix = 1 << bits;
//
//    //check if local data have outflowed the local memory
//    checkLocalMemOverflow(localSize * sizeof(int) * radix);
//
//    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record) * rLen, NULL, &status);
//    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int) * radix * itemNum, NULL, &status);
//
//    int argsNum = 0;
//
//    struct timeval start, end;
//
//    //partition according to the lower n bits (totalCountBits)
//    for(int shift = 0 ; shift < totalCountBits; shift += bits) {
//
//        argsNum = 0;
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(cl_mem), &d_R);
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &rLen);
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(cl_mem), &d_his);
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int) * radix * localSize, NULL);
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &bits);
//        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &shift);
//        checkErr(status, ERR_SET_ARGUMENTS);
//
//#ifdef PRINT_KERNEL
//        printExecutingKernel(createListHJKernel);
//#endif
//        gettimeofday(&start, NULL);
//        status = clEnqueueNDRangeKernel(info.currentQueue, createListHJKernel, 1, 0, global, mylocal, 0, 0, 0);
//        status = clFinish(info.currentQueue);
//        gettimeofday(&end, NULL);
//        totalTime += diffTime(end, start);
//        checkErr(status, ERR_EXEC_KERNEL);
//
//        int * his = new int[radix * itemNum];
//
//        //call the scan function
//        totalTime += scan(d_his, radix * itemNum, 1, info);
//
//        status = clEnqueueReadBuffer(info.currentQueue, d_his, CL_TRUE, 0, sizeof(int) * radix * itemNum, his, 0, 0, 0);
//        checkErr(status, ERR_READ_BUFFER);
//
//        argsNum = 0;
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_R);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_dest);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &rLen);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_his);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int) * radix * localSize, NULL);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &bits);
//        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &shift);
//        checkErr(status, ERR_SET_ARGUMENTS );
//
//#ifdef PRINT_KERNEL
//        printExecutingKernel(splitWithListHJKernel);
//#endif
//        gettimeofday(&start, NULL);
//        status = clEnqueueNDRangeKernel(info.currentQueue, splitWithListHJKernel, 1, 0, global, mylocal, 0, 0, 0);
//        status = clFinish(info.currentQueue);
//        gettimeofday(&end, NULL);
//        totalTime += diffTime(end, start);
//        checkErr(status, ERR_EXEC_KERNEL);
//
//        status = clEnqueueCopyBuffer(info.currentQueue, d_dest, d_R, 0, 0, sizeof(Record)*rLen, 0, 0, 0);
//        checkErr(status, ERR_COPY_BUFFER);
//    }
//
//    return totalTime;
//}

//testing: only count the number of outputs
double hashjoin(cl_mem d_R, int rLen, cl_mem d_S, int sLen, int &res_len, PlatInfo info)
{
    struct timeval start, end;
    double totalTime = 0;
    cl_int status;
    int bits = 13;          //grid size = 1024
    int buckets = (1<<bits);

    cl_mem d_R_partitioned = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_S_partitioned = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem r_start = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*buckets, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem s_start = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*buckets, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    gettimeofday(&start, NULL);
    split(d_R, d_R_partitioned, r_start, rLen, bits, info);
    gettimeofday(&end, NULL);
    double r_time = diffTime(end, start);

    gettimeofday(&start, NULL);
    split(d_S, d_S_partitioned, s_start, sLen, bits, info);
    gettimeofday(&end, NULL);
    double s_time = diffTime(end, start);

    gettimeofday(&start, NULL);
    probe(d_R_partitioned, d_S_partitioned, rLen, sLen, r_start, s_start, buckets, info, res_len);
    gettimeofday(&end, NULL);
    double probeTime = diffTime(end, start);

    totalTime += r_time;
    totalTime += s_time;
    totalTime += probeTime;

    std::cout<<"R partition time: "<<r_time<<" ms."<<std::endl;
    std::cout<<"S partition time: "<<s_time<<" ms."<<std::endl;
    std::cout<<"Probe time: "<<probeTime<<" ms."<<std::endl;

    return totalTime;
}

double probe(cl_mem d_R, cl_mem d_S, int r_len, int s_len, cl_mem start_R, cl_mem start_S, int buckets, PlatInfo info, int &res_len)
{
    cl_int status;
    int argsNum = 0;
    double totalTime = 0;
    int gridSize = buckets;              //Each work-group works for a bucket-bucket pair
    int localSize = 1024;
    int globalSize = localSize * gridSize;

    //kernel reading
    char hjPath[100] = PROJECT_ROOT;
    strcat(hjPath, "/Kernels/hjKernel.cl");
    std::string hjKerAddr = hjPath;

    char probeSource[100] = "probe";
    KernelProcessor hjReader(&hjKerAddr,1,info.context, "");
    cl_kernel probeKernel = hjReader.getKernel(probeSource);

    //memory allocation
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*globalSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

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
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &r_len);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(int), &s_len);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &start_R);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &start_S);
    status |= clSetKernelArg(probeKernel, argsNum++, sizeof(cl_mem), &d_out_num);
    status |= clSetKernelArg(probeKernel, argsNum++, 23*1024, NULL);    //16KB
    status |= clSetKernelArg(probeKernel, argsNum++, 23*1024, NULL);
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
//    std::cout<<"# Joined result:"<<h_out_num<<std::endl;

    return totalTime;
}
