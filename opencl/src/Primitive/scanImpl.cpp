//
//  scanImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
#include "DataDef.h"
using namespace std;

/*
 *  grid size should be equal to the # of computing units
 *  R: number of elements in registers in each work-item
 *  L: number of elememts in local memory
 */
double scan_fast(cl_mem &d_inout, int length, PlatInfo& info, int local_size, int grid_size, int R, int L)
{
    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    int local_size_log = log2(local_size);
    int tile_size = local_size * (R + L);
    int num_of_blocks = (length + tile_size - 1) / tile_size;


    //kernel reading
    char extra[500];        
    strcpy(extra, "-DREGISTERS=");
    char R_li[20];

    //calculating the demanding intermediate local memory size
    int lo_size;            //intermediate memory size
    int local_mem_size;      //actual memory size

    if (R==0 && L==0) {
        cerr<<"Parameter error. R and L can not be 0 at the same time."<<endl;
        return 1;
    }

    if (R==0)   lo_size = L * local_size;
    else        lo_size = (L+1)*local_size;

    if (lo_size > local_size * R)       local_mem_size = lo_size;
    else                                local_mem_size = local_size * R;

    int DR;
    if (R == 0) DR = 1;
    else        DR = R;
    my_itoa(DR, R_li, 10);       //transfer R to string
    strcat(extra, R_li);

    cl_kernel scanBlockKernel = KernelProcessor::getKernel("scanKernel.cl", "scan_fast", info.context, extra);

    //initialize the intermediate array
    int *h_inter = new int[num_of_blocks];
    for(int i = 0; i < num_of_blocks; i++) h_inter[i] = -1;

    cl_mem d_inter = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* num_of_blocks, NULL, &status);
    status = clEnqueueWriteBuffer(info.currentQueue, d_inter, CL_TRUE, 0, sizeof(int)*num_of_blocks, h_inter, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size * grid_size)};

    //help, for debug
//    cl_mem d_help = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*local_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int)*local_mem_size, NULL);    //local memory lo
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &lo_size);           //local mem size
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &num_of_blocks);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &R);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &L);
    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_inter);

//    status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_help);

    checkErr(status, ERR_SET_ARGUMENTS);
    
#ifdef PRINT_KERNEL
    printExecutingKernel(scanBlockKernel);
#endif
    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    totalTime = clEventTime(event);

//    int *h_help = new int[local_size];
//    status = clEnqueueReadBuffer(info.currentQueue, d_help, CL_TRUE, 0, sizeof(int)*local_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < local_size; i++) cout<<i<<' '<<h_help[i]<<endl;
//    cout<<endl;
//    delete[] h_help;
//    clReleaseMemObject(d_help);

    clReleaseMemObject(d_inter);
    if(h_inter) delete[] h_inter;

    return totalTime;
}

double scan_three_kernel(cl_mem &d_inout, unsigned length, PlatInfo &info, int local_size, int grid_size)
{
    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;
    int global_size = local_size * grid_size;

    const int reduce_ele_per_wi = (length + global_size -1)/global_size;
    const int scan_ele_per_wi = 14;      //temporary number

    //conpilation parameters
    char extra[500], para_reduce[20], para_scan[20];
    my_itoa(reduce_ele_per_wi, para_reduce, 10);       //transfer R to string
    my_itoa(scan_ele_per_wi, para_scan, 10);       //transfer R to string

    strcpy(extra, "-DREDUCE_ELE_PER_WI=");
    strcat(extra, para_reduce);
    strcat(extra, " -DSCAN_ELE_PER_WI=");
    strcat(extra, para_scan);
//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = KernelProcessor::getKernel("scan_rss_kernel.cl", "reduce", info.context, extra);

    size_t reduce_local[1] = {(size_t)local_size};
    size_t reduce_global[1] = {(size_t)(global_size)};

    cl_mem d_reduction = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    //help, for debug
//    cl_mem d_help = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int)*grid_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, reduce_kernel, 1, 0, reduce_global, reduce_local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

//-------------------------- Step 2: intermediate scan -----------------------------
    cl_kernel scan_small_kernel = KernelProcessor::getKernel("scan_rss_kernel.cl", "scan_exclusive_small", info.context,extra); //still need extra paras

    size_t scan_small_local[1] = {(size_t)local_size};
    size_t scan_small_global[1] = {(size_t)(local_size*1)};

    argsNum = 0;
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int), &grid_size);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int)*grid_size, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, scan_small_kernel, 1, 0, scan_small_global, scan_small_local, 0, NULL, &event); //single WG execution
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

//-------------------------- Step 3: final exclusive scan -----------------------------
    cl_kernel scan_kernel = KernelProcessor::getKernel("scan_rss_kernel.cl", "scan_exclusive", info.context, extra);

    int scan_len_per_wg = (length + grid_size - 1)/grid_size;

    size_t scan_local[1] = {(size_t)local_size};
    size_t scan_global[1] = {(size_t)(local_size*grid_size)};

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &scan_len_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int)*local_size*scan_ele_per_wi, NULL);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, scan_kernel, 1, 0, scan_global, scan_local, 0, NULL, &event);
    status = clFinish(info.currentQueue);

    status = clEnqueueNDRangeKernel(info.currentQueue, scan_kernel, 1, 0, scan_global, scan_local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

//    std::cout<<"local_size:"<<local_size<<'\t'<<"grid_size:"<<grid_size<<'\t'<<"ele_per_wg:"<<length/grid_size<<std::endl;
//    int *h_help = new int[grid_size];
//    status = clEnqueueReadBuffer(info.currentQueue, d_reduction, CL_TRUE, 0, sizeof(int)*grid_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < grid_size; i++) {
//        cout<<h_help[i]<<' ';
//    }
//    cout<<endl;
//    for(int i = 1; i < grid_size; i++) {
//        if (h_help[i] - h_help[i-1]!= 1024) {
//            cout<<i<<' '<<h_help[i]<<' '<<h_help[i-1]<<endl;
//        }
//    }
//    cout<<endl;
//    delete[] h_help;
    clReleaseMemObject(d_reduction);

    cout<<endl;
    cout<<"reduce time:"<<reduce_time<<" ms"<<endl;
    cout<<"scan small time:"<<scan_small_time<<" ms"<<endl;
    cout<<"scan exclusive time:"<<scan_time<<" ms"<<endl;

    return totalTime;
}

//single-threaded
double scan_three_kernel_single(cl_mem &d_inout, unsigned length, PlatInfo &info, int grid_size)
{
    double totalTime = 0.0f;
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;

    //single-threaded
    int local_size = 1;
    int len_per_wg = (length + grid_size - 1) / grid_size;

//-------------------------- Step 1: reduce -----------------------------
    cl_kernel reduce_kernel = KernelProcessor::getKernel("scan_rss_single_kernel.cl", "reduce_single", info.context);

    size_t local[1] = {(size_t)local_size};
    size_t global[1] = {(size_t)(local_size*grid_size)};

    cl_mem d_reduction = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*grid_size, NULL, &status);

    argsNum = 0;
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(reduce_kernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, reduce_kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double reduce_time = clEventTime(event);
    totalTime += reduce_time;

//-------------------------- Step 2: intermediate scan -----------------------------
    cl_kernel scan_small_kernel = KernelProcessor::getKernel("scan_rss_single_kernel.cl", "scan_no_offset_single", info.context); //still need extra paras

    size_t small_local[1] = {(size_t)(1)};
    size_t small_global[1] = {(size_t)(1)};

    argsNum = 0;
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_small_kernel, argsNum++, sizeof(int), &grid_size);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, scan_small_kernel, 1, 0, small_global, small_local, 0, NULL, &event);
    clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_small_time = clEventTime(event);
    totalTime += scan_small_time;

//-------------------------- Step 3: final exclusive scan  -----------------------------

    cl_kernel scan_kernel = KernelProcessor::getKernel("scan_rss_single_kernel.cl", "scan_with_offset_single", info.context);

    argsNum = 0;
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_inout);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(cl_mem), &d_reduction);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &len_per_wg);
    status |= clSetKernelArg(scan_kernel, argsNum++, sizeof(int), &length);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    status = clEnqueueNDRangeKernel(info.currentQueue, scan_kernel, 1, 0, global, local, 0, NULL, &event);
    status = clFinish(info.currentQueue);
    checkErr(status, ERR_EXEC_KERNEL);
    double scan_time = clEventTime(event);
    totalTime += scan_time;

//    std::cout<<"local_size:"<<local_size<<'\t'<<"grid_size:"<<grid_size<<'\t'<<"ele_per_wg:"<<length/grid_size<<std::endl;
//    int *h_help = new int[grid_size];
//    status = clEnqueueReadBuffer(info.currentQueue, d_reduction, CL_TRUE, 0, sizeof(int)*grid_size, h_help, 0, 0, 0);
//    cout<<endl;
//    for(int i = 0; i < grid_size; i++) {
//        cout<<h_help[i]<<' ';
//    }
//    cout<<endl;
//    delete[] h_help;

    clReleaseMemObject(d_reduction);
    return totalTime;
}

double scan(cl_mem &d_in, cl_mem &d_out, int length, PlatInfo& info, int local_size)
{
//    double totalTime = 0.0f;
//    cl_event event;
//    cl_int status = 0;
//    int argsNum = 0;
//
//    int element_per_block = local_size * SCAN_ELE_PER_THREAD;
//
//    //decide how many levels should we handle(at most 3 levels: 8192^3)
//    int first_level_block_num = (length + element_per_block - 1 )/ element_per_block;
//    int second_level_block_num = (first_level_block_num + element_per_block - 1) / element_per_block;
//    int third_level_block_num = (second_level_block_num + element_per_block - 1) / element_per_block;
//
//    size_t local[1] = {(size_t)local_size};
//    size_t first_global[1] = {(size_t)local_size * (size_t)first_level_block_num};
//    size_t second_global[1] = {(size_t)local_size * (size_t)second_level_block_num};
//
//    //length should be less than element_per_block^2
//    if(second_level_block_num > 1) {
//        std::cerr<<"dataSize too large for this block size."<<std::endl;
//        return 1;
//    }
//
//    //kernel reading
//    KernelProcessor scanReader(&scanKerAddr,1,info.context, extra);
//
//    cl_kernel reduce_kernel = KernelProcessor::getKernel("scan_rss_kernel.cl", "reduce", info.context);
//
//
//    cl_kernel scanBlockKernel = scanReader.getKernel(scanBlock);
//    cl_kernel scanAddBlockKernel = scanReader.getKernel(scanAddBlock);
//
//    int warpSize = SCAN_WARPSIZE;
//    int numOfWarps = local_size / warpSize;
//
//    if (first_level_block_num == 1) {      //length <= element_per_block, only 1 level is enough
//        int isWriteSum = 0;
//        argsNum = 0;
//
//        int *firstTempAfter = new int[length];
//
//        status = clEnqueueReadBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*length, firstTempAfter, 0, NULL, NULL);
//        checkErr(status, ERR_READ_BUFFER);
//
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_in);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);
//
//        checkErr(status, ERR_SET_ARGUMENTS);
//
//    #ifdef PRINT_KERNEL
//        printExecutingKernel(scanBlockKernel);
//    #endif
//        status = clFinish(info.currentQueue);
//
//        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, first_global, local, 0, NULL, &event);
//        clFlush(info.currentQueue);
//        checkErr(status, ERR_EXEC_KERNEL);
//        totalTime += clEventTime(event);
//
//        status = clEnqueueReadBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*length, firstTempAfter, 0, NULL, NULL);
//        checkErr(status, ERR_READ_BUFFER);
//    }
//    else if (second_level_block_num == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
//        cl_mem firstBlockSum = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*first_level_block_num, NULL, &status);   //first level block sum
//        checkErr(status, ERR_HOST_ALLOCATION);
//
//        int isWriteSum = 1;
//        argsNum = 0;
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &d_in);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &length);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);
//        checkErr(status, ERR_SET_ARGUMENTS);
//
//    #ifdef PRINT_KERNEL
//        printExecutingKernel(scanBlockKernel);
//    #endif
//        status = clFinish(info.currentQueue);
//
//        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, first_global, local, 0, NULL, &event);
//        clFlush(info.currentQueue);
//        checkErr(status, ERR_EXEC_KERNEL);
//        totalTime += clEventTime(event);
//
//        int *firstTemp = new int[first_level_block_num];
//        int *firstTempAfter = new int[first_level_block_num];
//
//        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*first_level_block_num, firstTemp, 0, NULL, NULL);
//        checkErr(status, ERR_READ_BUFFER);
//
//        isWriteSum = 0;
//        argsNum = 0;
//        isExclusive = 0;
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &first_level_block_num);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isExclusive);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &isWriteSum);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(cl_mem), NULL);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int), &numOfWarps);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE*SCAN_ELE_PER_THREAD, NULL);       //host can specify the size for local memory
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * SCAN_MAX_BLOCKSIZE, NULL);
//        status |= clSetKernelArg(scanBlockKernel, argsNum++, sizeof(int) * numOfWarps, NULL);
//        checkErr(status, ERR_SET_ARGUMENTS);
//
//    #ifdef PRINT_KERNEL
//        printExecutingKernel(scanBlockKernel);
//    #endif
//        status = clFinish(info.currentQueue);
//
//        status = clEnqueueNDRangeKernel(info.currentQueue, scanBlockKernel, 1, 0, second_global, local, 0, NULL, &event);
//        clFlush(info.currentQueue);
//        checkErr(status, ERR_EXEC_KERNEL);
//        totalTime += clEventTime(event);
//
//        status = clEnqueueReadBuffer(info.currentQueue, firstBlockSum, CL_TRUE, 0, sizeof(int)*first_level_block_num, firstTempAfter, 0, NULL, NULL);
//        checkErr(status, ERR_READ_BUFFER);
//
//        //add block
//        argsNum = 0;
//        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &d_in);
//        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(int), &length);
//        status |= clSetKernelArg(scanAddBlockKernel, argsNum++, sizeof(cl_mem), &firstBlockSum);
//        checkErr(status, ERR_SET_ARGUMENTS);
//
//    #ifdef PRINT_KERNEL
//        printExecutingKernel(scanAddBlockKernel);
//    #endif
//        status = clFinish(info.currentQueue);
//
//        status = clEnqueueNDRangeKernel(info.currentQueue, scanAddBlockKernel, 1, 0, first_global, local, 0, NULL, &event);
//        clFlush(info.currentQueue);
//        checkErr(status, ERR_EXEC_KERNEL);
//        totalTime += clEventTime(event);
//
//    }
//    return totalTime;
}

