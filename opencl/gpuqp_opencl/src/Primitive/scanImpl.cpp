//
//  scanImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "scanImpl.h"

/* cl_arr : the cl_mem object that needs to be scaned
 * isExclusive: 1 for exclusive
 *              0 for inclusive
 * return : processing time
 */
double scan(cl_mem &cl_arr, int num,int isExclusive, PlatInfo info, int localSize) {
    
    double totalTime = 0;
    
    int firstLevelBlockNum = ceil((double)num / (localSize*2));
    int secondLevelBlockNum = ceil((double)firstLevelBlockNum / (localSize*2));

    //set the local and global size
    size_t local[1] = {(size_t)localSize};
    size_t firstGlobal[1] = {(size_t)localSize * (size_t)firstLevelBlockNum};
    size_t secondGlobal[1] = {(size_t)localSize * (size_t)secondLevelBlockNum};
    size_t thirdGlobal[1] = {(size_t)localSize};
    
    cl_int status = 0;
    int argsNum = 0;
    uint tempSize = localSize * 2;
    
    //kernel reading
    char scanPath[100] = PROJECT_ROOT;
    strcat(scanPath, "/Kernels/scanKernel.cl");
    std::string scanKerAddr = scanPath;
    
    char prefixScanSource[100] = "prefixScan";
    char scanLargeSource[100] = "scanLargeArray";
    char addBlockSource[100] = "addBlock";
    
    KernelProcessor scanReader(&scanKerAddr,1,info.context);
    
    cl_kernel psKernel = scanReader.getKernel(prefixScanSource);
    cl_kernel largePsKernel = scanReader.getKernel(scanLargeSource);
    cl_kernel addBlockKernel = scanReader.getKernel(addBlockSource);
    
    struct timeval start, end;
    
    //temp memory objects
    cl_mem cl_firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*firstLevelBlockNum, NULL, &status);   //first level block sum
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem cl_secondBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)*secondLevelBlockNum, NULL, &status); //second level block sum
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //Scan the whole array
    argsNum = 0;
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(cl_mem), &cl_arr);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(uint), &num);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(int), &isExclusive);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(int)*localSize*2, NULL);       //host can specify the size for local memory
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(uint), &tempSize);       //host can specify the size for local memory
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(cl_mem), &cl_firstBlockSum);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(largePsKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, largePsKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime += diffTime(end, start);
    
    //Scan the blockSum
    isExclusive = 0;            //scan the blockSum should use inclusive
    argsNum = 0;
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(cl_mem), &cl_firstBlockSum);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(uint), &firstLevelBlockNum);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(int), &isExclusive);
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(int)*localSize*2, NULL);       //host can specify the size for local memory
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(uint), &tempSize);       //host can specify the size for local memory
    status |= clSetKernelArg(largePsKernel, argsNum++, sizeof(cl_mem), &cl_secondBlockSum);
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(largePsKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, largePsKernel, 1, 0, secondGlobal, local, 0, NULL, NULL);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime += diffTime(end, start);
    
    //Normal prefix scan the second block
    argsNum = 0;
    status |= clSetKernelArg(psKernel, argsNum++, sizeof(cl_mem), &cl_secondBlockSum);
    status |= clSetKernelArg(psKernel, argsNum++, sizeof(uint), &secondLevelBlockNum);
    status |= clSetKernelArg(psKernel, argsNum++, sizeof(int), &isExclusive);
    status |= clSetKernelArg(psKernel, argsNum++, sizeof(int)*localSize*2, NULL);       //host can specify the size for local memory
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(psKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, psKernel, 1, 0, thirdGlobal, local, 0, NULL, NULL);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime += diffTime(end, start);
    
    //Add the second block sum
    argsNum = 0;
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(cl_mem), &cl_firstBlockSum);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(uint), &firstLevelBlockNum);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(cl_mem), &cl_secondBlockSum);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(int)*localSize*2, NULL);       //host can specify the size for local memory
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(addBlockKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, addBlockKernel, 1, 0, secondGlobal, local, 0, NULL, NULL);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime += diffTime(end, start);
    
    //Add the first block sum
    argsNum = 0;
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(cl_mem), &cl_arr);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(uint), &num);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(cl_mem), &cl_firstBlockSum);
    status |= clSetKernelArg(addBlockKernel, argsNum++, sizeof(int)*localSize*2, NULL);       //host can specify the size for local memory
    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(addBlockKernel);
#endif
    gettimeofday(&start, NULL);
    status = clEnqueueNDRangeKernel(info.currentQueue, addBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
    status = clFinish(info.currentQueue);
    gettimeofday(&end, NULL);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime += diffTime(end, start);
    
    status = clReleaseMemObject(cl_firstBlockSum);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(cl_secondBlockSum);
    checkErr(status, ERR_RELEASE_MEM);
    
    return totalTime;
}
