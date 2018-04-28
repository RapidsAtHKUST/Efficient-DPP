//
//  TestBasic.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 6/14/16.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
using namespace std;

/*
 * Test scalar multiplication bandwidth
 */
void testMem(PlatInfo& info) {
    cl_int status = 0;
    cl_event event;
    int argsNum;
    int localSize = 1024;
    int gridSize = 262144;
    double mulTime = 0.0;
    int scalar = 13;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    //get the kernel
    cl_kernel mul_kernel = KernelProcessor::getKernel("memKernel.cl", "mem_mul_bandwidth", info.context);

    int len = localSize*gridSize;
    std::cout<<"Data size for read/write(multiplication test): "<<len<<" ("<<len*sizeof(int)*1.0/1024/1024<<"MB)"<<std::endl;

    //data initialization
    int *h_in = new int[len];
    for(int i = 0; i < len; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(int)*len, NULL, &status);
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(int), &scalar);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, mul_kernel, 1, 0, global, local, 0, 0, &event);  //kernel execution
        status = clFinish(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);

        //throw away the first result
        if (i != 0) mulTime += clEventTime(event);
    }
    mulTime /= (MEM_EXPR_TIME - 1);

    status = clFinish(info.currentQueue);
    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_in;

    //compute the bandwidth, including read and write
    double throughput = computeMem(len*2, sizeof(int), mulTime);
    cout<<"Time for multiplication: "<<mulTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput<<" GB/s"<<endl;
}

void testAccess(PlatInfo& info) {
    cl_event event;
    cl_int status = 0;
    int argsNum = 0;
    int localSize = 1024, gridSize = 8192;
    int repeat_max = 100;
    int length = localSize * gridSize * repeat_max;
    std::cout<<"Maximal data size: "<<length<<" ("<<length* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    //kernel reading
    cl_kernel mul_strided_kernel = KernelProcessor::getKernel("memKernel.cl", "mem_mul_strided", info.context);
    cl_kernel mul_coalesced_kernel = KernelProcessor::getKernel("memKernel.cl", "mem_mul_coalesced", info.context);
    cl_kernel mul_warped_kernel = KernelProcessor::getKernel("memKernel.cl", "mem_mul_strided_warpwise", info.context);

    //memory allocation
    int *h_in = new int[length];
    int *h_out = new int[length];
    for(int i = 0; i < length; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*length, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clFinish(info.currentQueue);

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    //set kernel arguments: mul_coalesced_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_strided_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_warped_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_warped_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_warped_kernel, argsNum++, sizeof(cl_mem), &d_out);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    //time recorder
    double *coalesced_time = new double[repeat_max+1];
    double *coalesced_throughput = new double[repeat_max+1];
    double *strided_time = new double[repeat_max+1];
    double *strided_throughput = new double[repeat_max+1];
    double *warped_time = new double[repeat_max+1];
    double *warped_throughput = new double[repeat_max+1];

    for(int i = 1; i <= repeat_max; i++) {
        coalesced_time[i] = 0.0;
        coalesced_throughput[i] = 0.0;
        strided_time[i] = 0.0;
        strided_throughput[i] = 0.0;
        warped_time[i] = 0.0;
        warped_throughput[i] = 0.0;
    }

    //coalesced
    cout<<"------------------ Coalesced Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < MEM_EXPR_TIME; i++) {
            status |= clSetKernelArg(mul_coalesced_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(info.currentQueue, mul_coalesced_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     coalesced_time[re] += tempTime;
        }
    }

    for(int re = 1; re <= repeat_max; re++) {
        coalesced_time[re] /= (MEM_EXPR_TIME - 1);
        assert(coalesced_time[re] > 0);
        coalesced_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), coalesced_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<coalesced_time[re]<<" ms\t"<<"Throughput: "<<coalesced_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    //strided
    cout<<"------------------ Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < MEM_EXPR_TIME; i++) {
            status |= clSetKernelArg(mul_strided_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(info.currentQueue, mul_strided_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     strided_time[re] += tempTime;
        }
    }

    //warpwise strided
    for(int re = 1; re <= repeat_max; re++) {
        strided_time[re] /= (MEM_EXPR_TIME - 1);
        assert(strided_time[re] > 0);
        strided_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), strided_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<strided_time[re]<<" ms\t"<<"Throughput: "<<strided_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    cout<<"------------------ Warpwise Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat_max; re++) {
        for(int i = 0; i < MEM_EXPR_TIME; i++) {
            status |= clSetKernelArg(mul_warped_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(info.currentQueue, mul_warped_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);

            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     warped_time[re] += tempTime;
        }
    }

    for(int re = 1; re <= repeat_max; re++) {
        warped_time[re] /= (MEM_EXPR_TIME - 1);
        assert(warped_time[re] > 0);
        warped_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(int), warped_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<","<<localSize*gridSize*(re)*1.0* sizeof(int)/1024/1024<<"MB)"
            <<" Time: "<<warped_time[re]<<" ms\t"<<"Throughput: "<<warped_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    status = clFinish(info.currentQueue);

    //for python formating
    cout<<endl;
    cout<<"strided_bandwidth = ["<<strided_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<strided_throughput[re];
    }
    cout<<"]"<<endl;

    cout<<"coalesced_bandwidth = ["<<coalesced_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<coalesced_throughput[re];
    }
    cout<<"]"<<endl;

    cout<<"mix_bandwidth = ["<<warped_throughput[1];
    for(int re = 2; re <= repeat_max; re++) {
        cout<<','<<warped_throughput[re];
    }
    cout<<"]"<<endl;

    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_in;
    delete [] h_out;

    delete[] coalesced_time;
    delete[] coalesced_throughput;
    delete[] strided_time;
    delete[] strided_throughput;
    delete[] warped_time;
    delete[] warped_throughput;

#ifndef SILENCE
    FUNC_END;
#endif
}

void testBarrier(
    float *fixedValues, PlatInfo& info , double& totalTime, double& percentage, int localSize, int gridSize) {

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    float *h_source_values = new float[localSize * gridSize];

    for(int i = 0; i < localSize * gridSize; i++) {
        h_source_values[i] = fixedValues[i];
    }

    //memory allocation
    cl_int status = 0;
    totalTime = 0;
    int argsNum = 0;

    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(float)*localSize * gridSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*localSize*gridSize, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    //call barrier
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    //kernel reading
    cl_kernel kernel_in = KernelProcessor::getKernel("barrierKernel.cl", "barrier_in", info.context);
    cl_kernel kernel_free = KernelProcessor::getKernel("barrierKernel.cl", "barrier_free", info.context);

    const int repeatTime = BARRIER_REPEAT_TIME;


    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel_in, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel_in, argsNum++, sizeof(int), &repeatTime);
    checkErr(status, ERR_SET_ARGUMENTS);

    argsNum = 0;
    status |= clSetKernelArg(kernel_free, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(kernel_free, argsNum++, sizeof(int), &repeatTime);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel_in);
    printExecutingKernel(kernel_free);
#endif

    double totalTime_in = 0.0;
    double totalTime_free = 0.0;

    cl_event event;
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    for(int i = 0; i < BARRIER_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_in, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_in = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_in += tempTime_in;

        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_free, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_free = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_free += tempTime_free;
    }    
    totalTime_in /= (BARRIER_EXPR_TIME - 1);
    totalTime_free /= (BARRIER_EXPR_TIME - 1);

    totalTime = totalTime_in - totalTime_free;
    percentage = totalTime / totalTime_in * 100;

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*localSize*gridSize, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    status = clReleaseMemObject(d_source_values);

    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;

#ifndef SILENCE
    SHOW_TIME(totalTime);
    FUNC_END;
#endif
}

void testAtomic(PlatInfo& info) {
           
    cl_int status;
    int args = 0, exper_time = 10;
    int local_size, grid_size;
    const int repeat_time = ATOMIC_REPEAT_TIME;
    double total_time_local = 0, total_time_global = 0;

    size_t local[1];
    size_t global[1];

//1. test local atomic throughput
    local_size = 1024;
    grid_size = 30;          //single-work-group
    local[0] = local_size;
    global[0] = local_size*grid_size;

    long value_local = 0;
    cl_mem d_local = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(long)*1, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_local, CL_TRUE, 0, sizeof(long)*1, &value_local, 0, 0, 0);

    cl_kernel kernel_local = KernelProcessor::getKernel("atomicKernel.cl", "atomic_local", info.context);
    status |= clSetKernelArg(kernel_local, args++, sizeof(cl_mem), &d_local);
    status |= clSetKernelArg(kernel_local, args++, sizeof(int), &repeat_time);
    checkErr(status, ERR_SET_ARGUMENTS);

    double totalTime_local = 0.0;

    cl_event event;
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    for(int i = 0; i < exper_time; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_local, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_local = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_local += tempTime_local;
    }
    totalTime_local /= (exper_time - 1);

    status = clEnqueueReadBuffer(info.currentQueue, d_local, CL_TRUE, 0, sizeof(long), &value_local, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    //checking
    long value_cal = local_size * grid_size * exper_time * ATOMIC_REPEAT_TIME;
    long data_size = (long)local_size * (long)grid_size * ATOMIC_REPEAT_TIME * sizeof(long);
    if (value_local == value_cal) {
        cout<<"Local test passes!"<<endl;
    }
    else {
        cout<<"Local test fails!"<<endl;
        cout<<value_local<<' '<<value_cal<<endl;
    }
    status = clFinish(info.currentQueue);
    status = clReleaseMemObject(d_local);
    checkErr(status, ERR_RELEASE_MEM);

    cout<<"Local size: "<<local_size<<'\t'
        <<"Grid size: "<<grid_size<<'\t'
        <<"Time: "<<totalTime_local<<" ms."
        <<"Bandwidth: "<<data_size/totalTime_local /1e6 <<" GB/s."<<endl;

//2. test global atomic throughput
    local_size = 1024;
    grid_size = 15;          //multi-work-group
    local[0] = local_size;
    global[0] = local_size*grid_size;

    long value_global = 0;
    cl_mem d_global = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(int)*1, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_global, CL_TRUE, 0, sizeof(int)*1, &value_global, 0, 0, 0);

    args = 0;
    cl_kernel kernel_global = KernelProcessor::getKernel("atomicKernel.cl", "atomic_global", info.context);
    status |= clSetKernelArg(kernel_global, args++, sizeof(cl_mem), &d_global);
    status |= clSetKernelArg(kernel_global, args++, sizeof(int), &repeat_time);
    checkErr(status, ERR_SET_ARGUMENTS);

    double totalTime_global = 0.0;

    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    for(int i = 0; i < exper_time; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, kernel_global, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime_global = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime_global += tempTime_global;
    }
    totalTime_global /= (exper_time - 1);

    status = clEnqueueReadBuffer(info.currentQueue, d_global, CL_TRUE, 0, sizeof(int), &value_global, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    //checking
    value_cal = (long)local_size * (long)grid_size * exper_time * ATOMIC_REPEAT_TIME;
    data_size = (long)local_size * (long)grid_size * ATOMIC_REPEAT_TIME * sizeof(int);
    if (value_global == value_cal) {
        cout<<"Global test passes!"<<endl;
    }
    else {
        cout<<"Global test fails!"<<endl;
        cout<<value_global<<' '<<value_cal<<endl;
    }
    status = clFinish(info.currentQueue);
    status = clReleaseMemObject(d_global);
    checkErr(status, ERR_RELEASE_MEM);

    cout<<"Local size: "<<local_size<<'\t'
        <<"Grid size: "<<grid_size<<'\t'
        <<"Time: "<<totalTime_global<<" ms."
        <<"Bandwidth: "<<data_size/totalTime_global /1e6 <<" GB/s."<<endl;
}

typedef unsigned long ptr_type;

//test cache and memory latency for one workitem
void testLatency(PlatInfo& info) {

    int numOfSizes = 9, numOfStrides = 18;
    int testSize[9] = {16,32,64,128,256,512,1024,2048,4096};   //KB
    int strides[18] = {8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072, 262144, 524288, 1048576};   //Byte

    double latencyTime[9][19] = {0.0};

    //kernel reading
    char *kerFileName = "latencyKernel.cl";
    cl_kernel kernel = KernelProcessor::getKernel(kerFileName, "latency", info.context);
    cl_kernel address_kernel = KernelProcessor::getKernel(kerFileName, "add_address", info.context);

    //test with different array size and strides
    for(int cs = 0; cs < numOfSizes; cs++) {
        int totalSize = testSize[cs] * 1024;    //change to Byte
        int totalNum = totalSize / sizeof(ptr_type); //change to # of tuples
        cout<<"-------------------------------------------"<<endl;
        cout<<"totalSize: "<<testSize[cs]<<"KB\t\t"<<"totalNum: "<<totalNum<<endl;

        for(int ss = 0; ss < numOfStrides; ss++) {
            //initialization
            int stride = strides[ss];
            int strideCount = stride / sizeof(ptr_type);
            cout<<"Stride: "<<stride<<'\t';

            //the extra place is used for storing the output
            ptr_type *h_source_values = new ptr_type[totalNum+1];        

            for(int i = 0; i < totalNum; i++) {
                h_source_values[i] = (( i + strideCount ) % totalNum) * sizeof(ptr_type);
            }

            //memory allocation
            cl_int status = 0;
            int argsNum = 0;

            cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(ptr_type)*(totalNum+1), NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(ptr_type)*(totalNum+1), h_source_values, 0, 0, 0);
            checkErr(status, ERR_WRITE_BUFFER);   

            //set kernel arguments for add_address kernel 
            argsNum = 0;
            status |= clSetKernelArg(address_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
            status |= clSetKernelArg(address_kernel, argsNum++, sizeof(int), &totalNum);
            checkErr(status, ERR_SET_ARGUMENTS);

            //set kernel arguments for latency test
            argsNum = 0;
            status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values);
            status |= clSetKernelArg(kernel, argsNum++, sizeof(int), &totalNum);
            checkErr(status, ERR_SET_ARGUMENTS);

            //for the address adding kernel
            size_t local[1] = {(size_t)1024};
            size_t global[1] = {(size_t)(1024 * 1024)};
            
            //one workgroup and one workitem per workgroup
            size_t latency_local[1] = {(size_t)1};
            size_t latency_global[1] = {(size_t)1};

            //launch the kernel
        #ifdef PRINT_KERNEL
            printExecutingKernel(kernel);
        #endif

            //adjust the device address
            status = clEnqueueNDRangeKernel(info.currentQueue, address_kernel, 1, 0, global, local, 0, 0, NULL);
            status = clFinish(info.currentQueue);

            //being latency test
            int experTime = 10;
            for(int i = 0; i < experTime; i++) {

                cl_event event;
                status = clEnqueueNDRangeKernel(info.currentQueue, kernel, 1, 0, latency_global, latency_local, 0, 0, &event);
                // clFlush(info.currentQueue);
                status = clFinish(info.currentQueue);

                checkErr(status, ERR_EXEC_KERNEL);
                double tempTime = clEventTime(event);

                //throw away the first result
                if (i != 0)     latencyTime[cs][ss] += tempTime;
            }    
            //repeated 2 000 000 times
            latencyTime[cs][ss] = latencyTime[cs][ss] * 1e6 / (experTime - 1) / 2000000 ;      //change to ns
            cout<<"latency time: "<<latencyTime[cs][ss]<<" ns." <<endl;

            status = clReleaseMemObject(d_source_values);
            checkErr(status, ERR_RELEASE_MEM);

            delete[] h_source_values;
        }
    }

    //for python
    for(int cs = 0; cs < numOfSizes; cs++) {
        int totalSize = testSize[cs] * 1024;    //change to Byte
        cout<<"latency_"<<testSize[cs]<<" = ["<<latencyTime[cs][0];

        for(int ss = 1; ss < numOfStrides; ss++) {
            //initialization
            cout<<','<<latencyTime[cs][ss];
        }
        cout<<"]"<<endl;
    }

    //for excel
    for(int cs = 0; cs < numOfSizes; cs++) {
        int totalSize = testSize[cs] * 1024;    //change to Byte
        int totalNum = totalSize / sizeof(ptr_type);
        cout<<"-------------------------------------------"<<endl;
        cout<<"totalSize: "<<testSize[cs]<<"KB\t\t"<<"totalNum: "<<totalNum<<endl;

        for(int ss = 0; ss < numOfStrides; ss++) {
            //initialization
            cout<<latencyTime[cs][ss]<<endl;
        }
    }
}