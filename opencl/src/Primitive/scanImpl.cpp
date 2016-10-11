//
//  scanImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
#include "dataDef.h"

using namespace std;

/*  grid size should be equal to the # of computing units
 *  R: number of elements in registers in each work-item
 *  L: number of elememts in local memory
 */
double scan_fast(cl_mem &d_source, int length, int isExclusive, PlatInfo& info, int localSize, int gridSize, int R, int L)
{
    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    int element_per_block = localSize * R + L;
    int num_of_blocks = (length + element_per_block - 1) / element_per_block;

    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    //kernel reading
    char scanPath[100] = PROJECT_ROOT;
    strcat(scanPath, "/Kernels/scanKernel.cl");
    std::string scanKerAddr = scanPath;

    char scanBlock[100] = "scan_fast";
    
    char extra[500];        
    strcpy(extra, "-DREGISTERS=");
    char R_li[20];

    //calculating the demanding intermediate local memory size
    int lo_size;            //intermediate memory size
    int localMem_size;      //actual memory size

    if (length < L)     {
        R = 0;      //no need to use registers
        L = length;
    }

    if ( R == 0 )   {   //no need to use register
        assert(L != 0);
        lo_size = L;
        localMem_size = lo_size;
    }
    else if ( L == 0 ) {    //no need to use local memory
        assert( R != 0);
        //need R * localSize local memory space
        lo_size = localSize;
        localMem_size = R * localSize;
    }
    else {          //use both local mem and registers
        //now length > L
        lo_size = localSize + L;
        localMem_size = (lo_size > R * localSize)? lo_size : (R*localSize);
    }
    
    int DR;
    if (R == 0) DR = 1;            //
    else        DR = R;
    my_itoa(DR, R_li, 10);       //transfer R to string
    strcat(extra, R_li);

    KernelProcessor scanReader(&scanKerAddr,1,info.context,extra);
    
    cl_kernel scanBlockKernel = scanReader.getKernel(scanBlock);

    //initialize the intermediate array
    int *h_inter = new int[num_of_blocks];
    cout<<"num of blocks:"<<num_of_blocks<<endl;
    for(int i = 0; i < num_of_blocks; i++) h_inter[i] = -1;

    cl_mem d_inter = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* num_of_blocks, NULL, &status);
    status = clEnqueueWriteBuffer(info.currentQueue, d_inter, CL_TRUE, 0, sizeof(int)*num_of_blocks, h_inter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    int warpNum = localSize / SCAN_WARPSIZE;

    argsNum = 0;
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int)*localMem_size, NULL);    //local memory lo
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &lo_size);           //local mem size

    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), NULL);    //gs

    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), NULL);    //gss
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int)*warpNum, NULL);    //help memory

    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &num_of_blocks);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &R);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &L);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_inter);

    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(scanBlockKernel);
#endif
    status = clFinish(info.currentQueue);

    status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);        
    totalTime += clEventTime(event);

    clReleaseMemObject(d_inter);
    delete[] h_inter;

    return totalTime;
}

double scan(cl_mem &d_source, int length, int isExclusive, PlatInfo& info, int localSize)
{
    double totalTime = 0.0f;
    cl_event event;

    cl_int status = 0;
    int argsNum = 0;

    int element_per_block = localSize * SCAN_ELE_PER_THREAD;
    //decide how many levels should we handle(at most 3 levels: 8192^3)
    int firstLevelBlockNum = (length + element_per_block - 1 )/ element_per_block;
    int secondLevelBlockNum = (firstLevelBlockNum + element_per_block - 1) / element_per_block;
    int thirdLevelBlockNum = (secondLevelBlockNum + element_per_block - 1) / element_per_block;

    size_t local[1] = {(size_t)localSize};
    size_t firstGlobal[1] = {(size_t)localSize * (size_t)firstLevelBlockNum};
    size_t secondGlobal[1] = {(size_t)localSize * (size_t)secondLevelBlockNum};
    size_t thirdGlobal[1] = {(size_t)localSize * (size_t)thirdLevelBlockNum};

    //length should be less than element_per_block^3
    if(thirdLevelBlockNum > 1) {
        std::cerr<<"dataSize too large for this block size."<<std::endl;    
        return 1;
    }

    char extra[500];        
    strcpy(extra, "-DREGISTERS=10");

    //kernel reading
    char scanPath[100] = PROJECT_ROOT;
    strcat(scanPath, "/Kernels/scanKernel.cl");
    std::string scanKerAddr = scanPath;

    char scanBlock[100] = "scan_block";
    char scanAddBlock[100] = "scan_addBlock";
    
    KernelProcessor scanReader(&scanKerAddr,1,info.context, extra);
    
    cl_kernel scanBlockKernel = scanReader.getKernel(scanBlock);
    cl_kernel scanAddBlockKernel = scanReader.getKernel(scanAddBlock);

    int warpSize = SCAN_WARPSIZE;
    int numOfWarps = localSize / warpSize;
    
    if (firstLevelBlockNum == 1) {      //length <= element_per_block, only 1 level is enough
        int isWriteSum = 0;
        argsNum = 0;

        int *firstTempAfter = new int[length];

        status = clEnqueueReadBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(int)*length, firstTempAfter, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL); 

        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, &event);
        clFlush(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);        
        totalTime += clEventTime(event);

        status = clEnqueueReadBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(int)*length, firstTempAfter, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);
    }
    else if (secondLevelBlockNum == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
        cl_mem firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*firstLevelBlockNum, NULL, &status);   //first level block sum
        checkErr(status, ERR_HOST_ALLOCATION);

        int isWriteSum = 1;
        argsNum = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);       
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, &event);
        clFlush(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);

        int *firstTemp = new int[firstLevelBlockNum];
        int *firstTempAfter = new int[firstLevelBlockNum];

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        isWriteSum = 0;
        argsNum = 0;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);       
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, secondGlobal, local, 0, NULL, &event);
        clFlush(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTempAfter, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);
        
        //add block
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, firstGlobal, local, 0, NULL, &event);
        clFlush(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);

    }
    else {                              //length <= element_per_block^3, 3 levels are enough
        cl_mem firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*firstLevelBlockNum, NULL, &status);   //first level block sum
        checkErr(status, ERR_HOST_ALLOCATION);
        cl_mem secondBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*secondLevelBlockNum, NULL, &status); //second level block sum
        checkErr(status, ERR_HOST_ALLOCATION);

        //1st
        int isWriteSum = 1;
        argsNum = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);       
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
#endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, &event);
        status = clFinish(info.currentQueue);

        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);
        cout<<"totalTime1:"<<totalTime<<endl;
        // int *firstTemp = new int[firstLevelBlockNum];
        // int *firstTempAfter = new int[firstLevelBlockNum];

        // status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTemp, 0, NULL, NULL);
        // checkErr(status, ERR_READ_BUFFER);

        argsNum = 0;
        isWriteSum = 1;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);       
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        clFlush(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, secondGlobal, local, 0, NULL, &event);
        status = clFinish(info.currentQueue);
        // clWaitForEvents(1,&event);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);
        cout<<"totalTime2:"<<totalTime<<endl;

        // status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTempAfter, 0, NULL, NULL);
        // checkErr(status, ERR_READ_BUFFER);

        //3rd
        int *secondTemp = new int[secondLevelBlockNum];

        status = clEnqueueReadBuffer(info.currentQueue, secondBlockSum, CL_TRUE, 0, sizeof(int)*secondLevelBlockNum, secondTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        isWriteSum = 0;
        argsNum = 0;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &secondLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);       
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, thirdGlobal, local, 0, NULL, &event);
        status = clFinish(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);

        totalTime += clEventTime(event);
        cout<<"totalTime3:"<<totalTime<<endl;

        status = clEnqueueReadBuffer(info.currentQueue, secondBlockSum, CL_TRUE, 0, sizeof(int)*secondLevelBlockNum, secondTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        //add block 1st
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, secondGlobal, local, 0, NULL, &event);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);
        cout<<"totalTime4:"<<totalTime<<endl;
        
        //add block 2nd
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        status = clFinish(info.currentQueue);

        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, firstGlobal, local, 0, NULL, &event);
        status = clFinish(info.currentQueue);

        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += clEventTime(event);
        cout<<"totalTime5:"<<totalTime<<endl;


    }
    return totalTime;
}

double scan_ble(cl_mem &d_source, int length, int isExclusive, PlatInfo& info, int localSize)
{
    double totalTime = 0.0f;

    cl_int status = 0;
    int argsNum = 0;

    int element_per_block = localSize * SCAN_ELE_PER_THREAD;
    //decide how many levels should we handle(at most 3 levels: 8192^3)
    int firstLevelBlockNum = (length + element_per_block - 1 )/ element_per_block;
    int secondLevelBlockNum = (firstLevelBlockNum + element_per_block - 1) / element_per_block;
    int thirdLevelBlockNum = (secondLevelBlockNum + element_per_block - 1) / element_per_block;

    size_t local[1] = {(size_t)localSize};
    size_t firstGlobal[1] = {(size_t)localSize * (size_t)firstLevelBlockNum};
    size_t secondGlobal[1] = {(size_t)localSize * (size_t)secondLevelBlockNum};
    size_t thirdGlobal[1] = {(size_t)localSize * (size_t)thirdLevelBlockNum};

    //length should be less than element_per_block^3
    if(thirdLevelBlockNum > 1) {
        std::cerr<<"data size too large for current block size."<<std::endl;
        return 0;
    }

    //kernel reading
    char scanPath[100] = PROJECT_ROOT;
    strcat(scanPath, "/Kernels/scanKernel.cl");
    std::string scanKerAddr = scanPath;

    char scanBlock[100] = "scan_ble_large";
    char scanAddBlock[100] = "scan_addBlock";
    
    KernelProcessor scanReader(&scanKerAddr,1,info.context);
    
    cl_kernel scanBlockKernel = scanReader.getKernel(scanBlock);
    cl_kernel scanAddBlockKernel = scanReader.getKernel(scanAddBlock);

    struct timeval start, end;
    
    if (firstLevelBlockNum == 1) {      //length <= element_per_block, only 1 level is enough
        int isWriteSum = 0;
        argsNum = 0;

        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
    }
    else if (secondLevelBlockNum == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
        cl_mem firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*firstLevelBlockNum, NULL, &status);   //first level block sum
        checkErr(status, ERR_HOST_ALLOCATION);

        int isWriteSum = 1;
        argsNum = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

        int *firstTemp = new int[firstLevelBlockNum];
        int *firstTempAfter = new int[firstLevelBlockNum];

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        isWriteSum = 0;
        argsNum = 0;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, secondGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTempAfter, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);
        
        //add block
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
    }
    else {                              //length <= element_per_block^3, 3 levels are enough
        cl_mem firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*firstLevelBlockNum, NULL, &status);   //first level block sum
        checkErr(status, ERR_HOST_ALLOCATION);
        cl_mem secondBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*secondLevelBlockNum, NULL, &status); //second level block sum
        checkErr(status, ERR_HOST_ALLOCATION);

        //1st
        int isWriteSum = 1;
        argsNum = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

        int *firstTemp = new int[firstLevelBlockNum];
        int *firstTempAfter = new int[firstLevelBlockNum];

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        argsNum = 0;
        isWriteSum = 1;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, secondGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*firstLevelBlockNum, firstTempAfter, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        //3rd
        int *secondTemp = new int[secondLevelBlockNum];

        status = clEnqueueReadBuffer(info.currentQueue, secondBlockSum, CL_TRUE, 0, sizeof(int)*secondLevelBlockNum, secondTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        isWriteSum = 0;
        argsNum = 0;
        isExclusive = 0;
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &secondLevelBlockNum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);      
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, thirdGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

        status = clEnqueueReadBuffer(info.currentQueue, secondBlockSum, CL_TRUE, 0, sizeof(int)*secondLevelBlockNum, secondTemp, 0, NULL, NULL);
        checkErr(status, ERR_READ_BUFFER);

        //add block 1st
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &firstLevelBlockNum);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &secondBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, secondGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);
        
        //add block 2nd
        argsNum = 0;
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &d_source);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &length);
        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
        checkErr(status, ERR_SET_ARGUMENTS);
        
    #ifdef PRINT_KERNEL
        printExecutingKernel(scanAddBlockKernel);
    #endif
        gettimeofday(&start, NULL);
        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, firstGlobal, local, 0, NULL, NULL);
        status = clFinish(info.currentQueue);
        gettimeofday(&end, NULL);
        checkErr(status, ERR_EXEC_KERNEL);
        totalTime += diffTime(end, start);

    }
    return totalTime;
}

//deprecated
/* cl_arr : the cl_mem object that needs to be scaned
 * isExclusive: 1 for exclusive
 *              0 for inclusive
 * return : processing time
 */
double scan_blelloch(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize) {
    
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
    status = clEnqueueNDRangeKernel(info.currentQueue, psKernel, 1, 0, thirdGlobal, local, 0, NULL, NULL);      //here grid size is actually 1!!
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
