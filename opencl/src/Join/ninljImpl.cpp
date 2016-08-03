//
//  ninljImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/6/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"

double ninlj(cl_mem& d_R, int rLen, cl_mem& d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize, int gridSize)
{
    double totalTime = 0;
    
    //number of result joined records
    int itemNum = localSize * gridSize;
    
    int locald_S_Length = 0;                                //actual number of records in a block local memory
    int pass = 0;
    
    int calLocalSLength = ceil(1.0 * sLen / gridSize);      //calculate how many records should one block store in local memory
    if (calLocalSLength * sizeof(Record) > MAX_LOCAL_MEM_SIZE) {
        locald_S_Length = MAX_LOCAL_MEM_SIZE / sizeof(Record);
        pass = ceil( sLen * 1.0 / ( gridSize * locald_S_Length ) );
    }
    else {
        locald_S_Length = calLocalSLength;
        pass = 1;
    }

    int countNum = itemNum * pass;
    uint tempCount = 0;
    
    cl_int status = 0;
    int argsNum = 0;
    int totalResNum = 0;
    
    //kernel reading
    char ninljPath[100] = PROJECT_ROOT;
    strcat(ninljPath, "/Kernels/ninljKernel.cl");
    std::string ninljKerAddr = ninljPath;
    
    char countMatchSource[100] = "countMatch";
    char writeMatchSource[100] = "writeMatch";
    
    KernelProcessor ninljReader(&ninljKerAddr,1,info.context);
    
    cl_kernel countMatchKernel = ninljReader.getKernel(countMatchSource);
    cl_kernel writeMatchKernel = ninljReader.getKernel(writeMatchSource);
    
    //memory allocation
    cl_mem d_count = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*countNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    struct timeval start, end;
    
    for(int tempPass = 0; tempPass < pass; tempPass++) {
        //set kernel arguments

        argsNum = 0;
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &rLen);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &sLen);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(Record)*locald_S_Length, NULL);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &locald_S_Length);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(uint)*localSize, NULL);
        status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &tempPass);
        checkErr(status, ERR_SET_ARGUMENTS);
    
        //launch the kernel
#ifdef PRINT_KERNEL
        printExecutingKernel(countMatchKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, countMatchKernel, 1, 0, global, mylocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        
        totalTime += diffTime(end, start);
        checkErr(status, ERR_EXEC_KERNEL);
    }
    
    //get the last num
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE, sizeof(uint)*(countNum-1), sizeof(uint), &tempCount, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    
    //scan
    totalTime += scan(d_count, countNum, 1, info);
    
    //get the last number
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE,sizeof(uint)*(countNum-1), sizeof(uint), &tempCount, 0, NULL, NULL);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    oLen = totalResNum;
    
    if (totalResNum == 0) {
        return totalTime;
    }
    
    d_Out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*totalResNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //write count
    for(int tempPass = 0; tempPass < pass; tempPass++) {
        //set kernel arguments
        argsNum = 0;
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &rLen);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &sLen);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_Out);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(Record)*locald_S_Length, NULL);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &locald_S_Length);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(uint)*localSize, NULL);
        status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &tempPass);
        checkErr(status, ERR_SET_ARGUMENTS);
        
        //launch the kernel
#ifdef PRINT_KERNEL
        printExecutingKernel(writeMatchKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, writeMatchKernel, 1, 0, global, mylocal, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        totalTime += diffTime(end, start);
        checkErr(status, ERR_EXEC_KERNEL);
    }

    return  totalTime;
}