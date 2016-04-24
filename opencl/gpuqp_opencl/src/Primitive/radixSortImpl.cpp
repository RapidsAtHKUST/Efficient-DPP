//
//  radixSortImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/6/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "radixSortImpl.h"
#include "scanImpl.h"
#include "scatterImpl.h"
#include "dataDef.h"

double radixSort(
#ifdef RECORDS
    cl_mem& d_source_keys, bool isRecords,
#endif
    cl_mem& d_source_values,
    int length, PlatInfo info) {
    
    double totalTime = 0;
    
    int gridSize = (length + REDUCE_BLOCK_SIZE * REDUCE_ELE_PER_THREAD- 1) / (REDUCE_BLOCK_SIZE * REDUCE_ELE_PER_THREAD);
    
    cl_int status;
    int argsNum = 0;

    uint hisSize = REDUCE_BLOCK_SIZE * gridSize * SORT_RADIX;
    uint isExclusive = 1;
    
    //kernel reading
    char sortPath[100] = PROJECT_ROOT;
    strcat(sortPath, "/Kernels/radixSortKernel.cl");
    std::string sortKerAddr = sortPath;
    
    // char countHisSource[100] = "countHis";
    // char writeHisSource[100] = "writeHis";
    char radixReduceSource[100] = "radix_reduce";
    char radixScatterSource[100] = "radix_scatter";

    KernelProcessor sortReader(&sortKerAddr,1,info.context);
    
    // cl_kernel countHisKernel = sortReader.getKernel(countHisSource);
    // cl_kernel writeHisKernel = sortReader.getKernel(writeHisSource);
    cl_kernel radixReduceKernel = sortReader.getKernel(radixReduceSource);
    cl_kernel radixScatterKernel = sortReader.getKernel(radixScatterSource);

    size_t reduce_localSize[1] = {(size_t)REDUCE_BLOCK_SIZE};
    size_t reduce_globalSize[1] = {(size_t)(REDUCE_BLOCK_SIZE * gridSize)};

    size_t scatter_localSize[1] = {(size_t)SCATTER_BLOCK_SIZE};
    size_t scatter_globalSize[1] = {(size_t)((gridSize+SCATTER_TILES_PER_BLOCK-1)/SCATTER_TILES_PER_BLOCK) * SCATTER_BLOCK_SIZE};

    struct timeval start, end;
    
    cl_mem d_temp_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
#ifdef RECORDS    
    cl_mem d_temp_keys = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION); 
#endif
    cl_mem d_histogram = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(uint)* hisSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    for(uint shiftBits = 0; shiftBits < sizeof(int) * 8; shiftBits += SORT_BITS) {
        
        //data preparation
        argsNum = 0;
        status |= clSetKernelArg(radixReduceKernel, argsNum++, sizeof(cl_mem), &d_source_values);
        status |= clSetKernelArg(radixReduceKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(radixReduceKernel, argsNum++, sizeof(cl_mem), &d_histogram);
        status |= clSetKernelArg(radixReduceKernel, argsNum++, sizeof(uint), &shiftBits);
        status |= clSetKernelArg(radixReduceKernel, argsNum++, sizeof(uint)*SORT_RADIX*REDUCE_BLOCK_SIZE, NULL);
        checkErr(status, ERR_SET_ARGUMENTS);
        
#ifdef PRINT_KERNEL
        printExecutingKernel(radixReduceKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, radixReduceKernel, 1, 0, reduce_globalSize, reduce_localSize, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
        
        totalTime += scan(d_histogram,hisSize,1,info);
        
        argsNum = 0;
#ifdef RECORDS
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(cl_mem), &d_source_keys);
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(cl_mem), &d_temp_keys);
#endif
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(cl_mem), &d_source_values);
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(cl_mem), &d_temp_values);

        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(uint), &length);
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(int), &gridSize);

        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(cl_mem), &d_histogram);
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(uint), &shiftBits);
        status |= clSetKernelArg(radixScatterKernel, argsNum++, sizeof(ScatterData)*SCATTER_TILES_PER_BLOCK, NULL);
        checkErr(status, ERR_SET_ARGUMENTS);
        
#ifdef PRINT_KERNEL
        printExecutingKernel(radixScatterKernel);
#endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, radixScatterKernel, 1, 0, scatter_globalSize, scatter_localSize, 0, 0, 0);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
        
        //copy the buffer for another loop
        status = clEnqueueCopyBuffer(info.currentQueue, d_temp_values, d_source_values, 0, 0, sizeof(int)*length,0 , 0, 0);
#ifdef RECORDS
        status = clEnqueueCopyBuffer(info.currentQueue, d_temp_keys, d_source_keys, 0, 0, sizeof(int)*length,0 , 0, 0);
#endif
        checkErr(status, ERR_COPY_BUFFER);
    }
    
    status = clReleaseMemObject(d_temp_values);
    checkErr(status, ERR_RELEASE_MEM);
#ifdef RECORDS
    status = clReleaseMemObject(d_temp_values);
    checkErr(status, ERR_RELEASE_MEM);
#endif
    status = clReleaseMemObject(d_histogram);
    checkErr(status, ERR_RELEASE_MEM);

    return totalTime;
}
