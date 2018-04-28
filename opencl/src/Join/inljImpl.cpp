//
//  inljImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/14/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
#include "CSSTree.h"

double inlj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, CSS_Tree_Info treeInfo, PlatInfo info, int localSize, int gridSize)
{
    double totalTime = 0;
    
    int itemNum = localSize * gridSize;
    
    if (treeInfo.CSS_length * sizeof(int) > MAX_LOCAL_MEM_SIZE) {
        std::cerr<<ERR_LOCAL_MEM_OVERFLOW<<std::endl;
    }
    
    cl_int status = 0;
    int argsNum = 0;
    int totalResNum = 0;
    uint tempCount = 0;
    
    //kernel reading
    cl_kernel countMatchKernel = KernelProcessor::getKernel("inljKernel.cl", "countMatch", info.context);
    cl_kernel writeMatchKernel = KernelProcessor::getKernel("inljKernel.cl", "writeMatch", info.context);

    //memory allocation
    cl_mem d_count = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*itemNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //set work group and NDRange sizes
    size_t mylocal[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    struct timeval start, end;
    
    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(cl_mem), &treeInfo.d_CSS);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int) * treeInfo.CSS_length, NULL);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &treeInfo.mPart);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &treeInfo.numOfInternalNodes);
    status |= clSetKernelArg(countMatchKernel, argsNum++, sizeof(int), &treeInfo.mark);
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
    
    //get the last num
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE, sizeof(uint)*(itemNum-1), sizeof(uint), &tempCount, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    
    totalResNum += tempCount;
    
    //scan 
    totalTime += scan(d_count, itemNum, 1, info);

    //get the last number
    status = clEnqueueReadBuffer(info.currentQueue, d_count, CL_TRUE,sizeof(uint)*(itemNum-1), sizeof(uint), &tempCount, 0, NULL, NULL);
    checkErr(status, ERR_READ_BUFFER);

    totalResNum += tempCount;
    oLen = totalResNum;
    
    if (totalResNum == 0) {
        return totalTime;
    }

    d_Out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*totalResNum, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //write count----------------------------------------------------------------------------------------------------------------

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_R);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &rLen);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_S);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &sLen);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_Out);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &d_count);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(cl_mem), &treeInfo.d_CSS);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int) * treeInfo.CSS_length, NULL);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &treeInfo.mPart);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &treeInfo.numOfInternalNodes);
    status |= clSetKernelArg(writeMatchKernel, argsNum++, sizeof(int), &treeInfo.mark);
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
    
    return  totalTime;
}