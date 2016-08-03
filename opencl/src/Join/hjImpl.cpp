//
//  hjImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/16/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

// totalCountBits: partition bits
double partitionHJ(cl_mem& d_R, int rLen,int totalCountBits, PlatInfo info, int localSize, int gridSize) {
    
    double totalTime = 0;
    
    cl_int status;
    int itemNum = gridSize * localSize;
    
    char splitPath[100] = PROJECT_ROOT;
    strcat(splitPath, "/Kernels/splitKernel.cl");
    std::string splitKerAddr = splitPath;

    char hjPath[100] = PROJECT_ROOT;
    strcat(hjPath, "/Kernels/hjKernel.cl");
    std::string hjKerAddr = hjPath;
    
    char createListHJSource[100] = "createListHJ";
    char splitWithListHJSource[100] = "splitWithListHJ";
    
    KernelProcessor splitReader(&splitKerAddr,1,info.context);
    KernelProcessor hjReader(&hjKerAddr,1,info.context);
    
    cl_kernel createListHJKernel = splitReader.getKernel(createListHJSource);
    cl_kernel splitWithListHJKernel = splitReader.getKernel(splitWithListHJSource);
    
    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    int bits = 4;
    int radix = 1 << bits;
    
    //check if local data have outflowed the local memory
    checkLocalMemOverflow(localSize * sizeof(int) * radix);
    
    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record) * rLen, NULL, &status);
    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int) * radix * itemNum, NULL, &status);
    
    int argsNum = 0;
    
    struct timeval start, end;
    
    //partition according to the lower n bits (totalCountBits)
    for(int shift = 0 ; shift < totalCountBits; shift += bits) {
        
        argsNum = 0;
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(cl_mem), &d_R);
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &rLen);
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int) * radix * localSize, NULL);
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &bits);
        status |= clSetKernelArg(createListHJKernel, argsNum++, sizeof(int), &shift);
        checkErr(status, ERR_SET_ARGUMENTS);
        
#ifdef PRINT_KERNEL
        printExecutingKernel(createListHJKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, createListHJKernel, 1, 0, global, mylocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        totalTime += diffTime(end, start);
        checkErr(status, ERR_EXEC_KERNEL);
        
        int * his = new int[radix * itemNum];
        
        //call the scan function
        totalTime += scan(d_his, radix * itemNum, 1, info);
        
        status = clEnqueueReadBuffer(info.currentQueue, d_his, CL_TRUE, 0, sizeof(int) * radix * itemNum, his, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        
        argsNum = 0;
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_R);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_dest);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &rLen);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(cl_mem), &d_his);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int) * radix * localSize, NULL);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &bits);
        status |= clSetKernelArg(splitWithListHJKernel, argsNum++, sizeof(int), &shift);
        checkErr(status, ERR_SET_ARGUMENTS );
        
#ifdef PRINT_KERNEL
        printExecutingKernel(splitWithListHJKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, splitWithListHJKernel, 1, 0, global, mylocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        totalTime += diffTime(end, start);
        checkErr(status, ERR_EXEC_KERNEL);
        
        status = clEnqueueCopyBuffer(info.currentQueue, d_dest, d_R, 0, 0, sizeof(Record)*rLen, 0, 0, 0);
        checkErr(status, ERR_COPY_BUFFER);
    }

    return totalTime;
}

// totalCountBits: partition bits
double hj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int totalCountBits, int localSize)
{
    double totalTime = 0;
    int blockNum = 1 << totalCountBits;              //2^totalCountBits blocks

    int itemNum = BLOCKSIZE * blockNum;
    
    cl_int status;
    int argsNum = 0;
    int localMaxNum = ceil( 1.0 * MAX_LOCAL_MEM_SIZE / sizeof(Record));
    int totalResNum = 0;
    int tempCount = 0;
    
    //kernel reading
    char hjPath[100] = PROJECT_ROOT;
    strcat(hjPath, "/Kernels/hjKernel.cl");
    std::string hjKerAddr = hjPath;
    
    char matchCountSource[100] = "matchCount";
    char matchWriteSource[100] = "matchWrite";
    
    KernelProcessor hjReader(&hjKerAddr,1,info.context);

    cl_kernel matchCountKernel = hjReader.getKernel(matchCountSource);
    cl_kernel matchWriteKernel = hjReader.getKernel(matchWriteSource);
    
    //memory allocation
    cl_mem d_his = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*itemNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)BLOCKSIZE};
    size_t global[1] = {(size_t)(BLOCKSIZE * blockNum)};
    
    struct timeval start, end;
    
    argsNum = 0;
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(int), &totalCountBits);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(int), &localMaxNum);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(Record)*localMaxNum, NULL);
    status |= clSetKernelArg(matchCountKernel, argsNum++, sizeof(int)*BLOCKSIZE, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

#ifdef PRINT_KERNEL
    printExecutingKernel(matchCountKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, matchCountKernel, 1, 0, global, mylocal, 0, 0, 0 );
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);

    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    //get the last num
    status = clEnqueueReadBuffer(info.currentQueue, d_his, CL_TRUE, sizeof(int)*(itemNum-1), sizeof(int), &tempCount, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;

    //scan the histogram
    totalTime += scan(d_his, itemNum, 1, info);
    
    status = clEnqueueReadBuffer(info.currentQueue, d_his, CL_TRUE, sizeof(int)*(itemNum-1), sizeof(int), &tempCount, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    oLen = totalResNum;
    
    //generate the result records
    if (totalResNum == 0) {
        return totalTime;
    }
    
    d_Out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*totalResNum, NULL, &status);
    
    argsNum = 0;
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(cl_mem), &d_Out);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(int), &totalCountBits);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(cl_mem), &d_his);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(int), &localMaxNum);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(Record)*localMaxNum, NULL);
    status |= clSetKernelArg(matchWriteKernel, argsNum++, sizeof(int)*BLOCKSIZE, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(matchWriteKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, matchWriteKernel, 1, 0, global, mylocal, 0, 0, 0 );
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    totalTime += diffTime(end, start);
    checkErr(status, ERR_EXEC_KERNEL);
    
    status = clReleaseMemObject(d_his);
    checkErr(status, ERR_RELEASE_MEM);
    
    return totalTime;
}
