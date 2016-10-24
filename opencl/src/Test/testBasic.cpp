//
//  TestBasic.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 6/14/16.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
using namespace std;

//for instruction throughput VPU test
template<typename T>
void testVPU(T *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize) {
    
    //total length
    int length = localSize * gridSize;

    if (basicSize != 1 && basicSize != 2 &&
        basicSize != 4 && basicSize != 8 && basicSize != 16) {
        cout<<"wrong basicSize: "<<basicSize<<endl;
        std::cerr<<"Wrong parameter for basicSize."<<std::endl;
        exit(1);
    }

    bool res = true;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
#endif

    T *h_source_values = new T[length];  

    for(int i = 0; i < length; i++) {
        h_source_values[i] = fixedValues[i];
    }
    
    cl_int status = 0;
    totalTime = 0.0;
    int repeatTime = VPU_REPEAT_TIME;

    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(T)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    char extra[500];
    if (sizeof(T) == sizeof(int))         
        strcpy(extra, "-DTYPE=int -DTYPE2=int2 -DTYPE4=int4 -DTYPE8=int8 -DTYPE16=int16");
    else if (sizeof(T) == sizeof(double))
        strcpy(extra, "-DTYPE=double -DTYPE2=double2 -DTYPE4=double4 -DTYPE8=double8 -DTYPE16=double16");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/vpuKernel.cl");
    std::string kerAddr = path;
    
    char kerName[100] = "vpu";
    char basicSizeName[20] = "";
    my_itoa(basicSize, basicSizeName, 10);
    strcat(kerName, basicSizeName);
    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel vpuKernel = reader.getKernel(kerName);

    //set kernel arguments
    int argsNum = 0;
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(vpuKernel, argsNum++, sizeof(int), &repeatTime);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(vpuKernel);
#endif

    cl_event event;
    status = clFinish(info.currentQueue);

    for(int i = 0; i < VPU_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, vpuKernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     totalTime += tempTime;
    }    
    totalTime /= (VPU_EXPR_TIME - 1);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(T)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    for(int i = 0; i < length; i++) cout<<h_source_values[i]<<' ';
    cout<<endl;

    status = clFinish(info.currentQueue);

    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;

#ifndef SILENCE
    FUNC_END;
#endif
}

//for memory read bandwidth test
template<typename T>
void testMem(PlatInfo& info , const int localSize, const int gridSize, double& readTime, double& writeTime, double& mulTime, double& addTime, int repeat) {

    cl_int status = 0;
    int argsNum = 0;
    
    int input_length_read = localSize / 2 * gridSize / 2 * repeat;  //shrink the localsize and gridsize

    //for write and mul, no need to repeat
    int input_length_others = localSize * gridSize * 2; 
    int output_length = localSize * gridSize * 2;

    std::cout<<"Input data size(read): "<<input_length_read<<std::endl;
    std::cout<<"Input data size(write, add & mul): "<<input_length_others<<std::endl;
    std::cout<<"Output data size: "<<output_length<<std::endl;

    assert(input_length_read > 0);
    assert(input_length_others > 0);
    assert(output_length > 0);

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
#endif

    T *h_source_values = new T[input_length_read];  
    T *h_source_values_2 = new T[input_length_read];  
    T *h_dest_values = new T[output_length];
    
    for(int i = 0; i < input_length_read; i++) {
        h_source_values[i] = rand() % 10000;
        h_source_values_2[i] = rand() % 10000;
    }

    readTime = 0.0; writeTime = 0.0; mulTime = 0.0; addTime = 0.0;

    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*input_length_read, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(T)*input_length_read, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*output_length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    char extra[500];
    if (sizeof(T) == sizeof(int))         strcpy(extra, "-DTYPE=int -DTYPE2=int2");
    else if (sizeof(T) == sizeof(double))   strcpy(extra, "-DTYPE=double -DTYPE2=double2");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;

    char read_kerName[100] = "mem_read";
    char write_kerName[100] = "mem_write";
    char mul_kerName[100] = "mem_mul";
    char add_kerName[100] = "mem_add";


    //get the kernel
    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel read_kernel = reader.getKernel(read_kerName);
    cl_kernel mul_kernel = reader.getKernel(mul_kerName);
    cl_kernel write_kernel = reader.getKernel(write_kerName);
    cl_kernel add_kernel = reader.getKernel(add_kerName);


    //set kernel arguments: read_kernel
    argsNum = 0;
    status |= clSetKernelArg(read_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(read_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(localSize/2)};
    size_t global[1] = {(size_t)(localSize * gridSize / 4)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel);
#endif

    cl_event event;
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);
    
    //executing read
    for(int i = 0; i < MEM_EXPR_TIME; i++) {

        status = clEnqueueNDRangeKernel(info.currentQueue, read_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     readTime += tempTime;
    }   
    //finish read test, free the input space
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*output_length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue); 

    //for write and mul test -------------------------------
    cl_mem d_source_values_1 = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*input_length_others, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_1, CL_TRUE, 0, sizeof(T)*input_length_others, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    cl_mem d_source_values_2 = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*input_length_others, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_2, CL_TRUE, 0, sizeof(T)*input_length_others, h_source_values_2, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    //set kernel arguments: write_kernel
    argsNum = 0;
    status |= clSetKernelArg(write_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: add_kernel
    argsNum = 0;
    status |= clSetKernelArg(add_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(add_kernel, argsNum++, sizeof(cl_mem), &d_source_values_2);
    status |= clSetKernelArg(add_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);

    local[0] = {(size_t)localSize};
    global[0] = {(size_t)(localSize * gridSize)};

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, write_kernel, 1, 0, global, local, 0, 0, &event);
        // clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     writeTime += tempTime;
    }    

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, mul_kernel, 1, 0, global, local, 0, 0, &event);
        // clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     mulTime += tempTime;
    }    

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, add_kernel, 1, 0, global, local, 0, 0, &event);
        // clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     addTime += tempTime;
    }  

    readTime /= ((MEM_EXPR_TIME - 1)*repeat);
    writeTime /= (MEM_EXPR_TIME - 1);
    mulTime /= (MEM_EXPR_TIME - 1);
    addTime /= (MEM_EXPR_TIME - 1);

    //memory written back for not being optimized out
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*output_length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);                                        

    status = clReleaseMemObject(d_source_values_1);
    status = clReleaseMemObject(d_source_values_2);
    status = clReleaseMemObject(d_dest_values);
    status = clReleaseMemObject(d_source_values);

    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    delete [] h_source_values_2;
    delete [] h_dest_values;

#ifndef SILENCE
    FUNC_END;
#endif
}

template<typename T>
void testAccess(PlatInfo& info , const int localSize, const int gridSize, int repeat) {

    cl_int status = 0;
    int argsNum = 0;
    
    int length = localSize * gridSize * repeat;

    std::cout<<"Data size: "<<length<<std::endl;
    assert(length > 0);

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    T *h_source_values = new T[length];  
    T *h_dest_values = new T[length];

    for(int i = 0; i < length; i++) {
        h_source_values[i] = (T)i;
    }

    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(T)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    char extra[500];
    if (sizeof(T) == sizeof(int))         strcpy(extra, "-DTYPE=int -DTYPE2=int2");
    else if (sizeof(T) == sizeof(double))   strcpy(extra, "-DTYPE=double -DTYPE2=double2");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;

    char mul_strided_kerName[100] = "mem_mul_strided";
    char mul_coalesced_kerName[100] = "mem_mul_coalesced";
    char mul_mix_kerName[100] = "mem_mul_strided_warpwise";

    //get the kernel
    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel mul_strided_kernel = reader.getKernel(mul_strided_kerName);
     cl_kernel mul_coalesced_kernel = reader.getKernel(mul_coalesced_kerName);
    cl_kernel mul_mix_kernel = reader.getKernel(mul_mix_kerName);

    //set kernel arguments: mul_coalesced_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_strided_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_mix_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_mix_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(mul_mix_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};
    
    //launch the kernel
#ifdef PRINT_KERNEL
    printExecutingKernel(kernel);
#endif

    cl_event event;
    clFlush(info.currentQueue);
    status = clFinish(info.currentQueue);

    //executing mul with coalesced and strided manner
    double *coalesced_time = new double[repeat+1];
    double *coalesced_throughput = new double[repeat+1];
    double *strided_time = new double[repeat+1];
    double *strided_throughput = new double[repeat+1];

    double *mix_time = new double[repeat+1];
    double *mix_throughput = new double[repeat+1];

    for(int i = 1; i <= repeat; i++) {
        coalesced_time[i] = 0.0;
        coalesced_throughput[i] = 0.0;
        strided_time[i] = 0.0;
        strided_throughput[i] = 0.0;
        mix_time[i] = 0.0;
        mix_throughput[i] = 0.0;
    }

    //coalesced
    cout<<"------------------ Coalesced Access ------------------"<<endl;
    for(int re = 1; re <= repeat; re++) {
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

    for(int re = 1; re <= repeat; re++) {
        coalesced_time[re] /= (MEM_EXPR_TIME - 1);
        assert(coalesced_time[re] > 0);
        coalesced_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(T), coalesced_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<')'
            <<" Time: "<<coalesced_time[re]<<" ms\t"<<"Throughput: "<<coalesced_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    //strided
    cout<<"------------------ Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat; re++) {
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
        // status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*length, h_dest_values, 0, 0, 0);
        // checkErr(status, ERR_READ_BUFFER);

        // for(int i = 0; i < length; i++) {
        //     cout<<h_source_values_1[i]<<' '<<h_dest_values[i]<<endl;
        // }

    }

    //warpwise strided
    for(int re = 1; re <= repeat; re++) {
        strided_time[re] /= (MEM_EXPR_TIME - 1);
        assert(strided_time[re] > 0);
        strided_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(T), strided_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<')'
            <<" Time: "<<strided_time[re]<<" ms\t"<<"Throughput: "<<strided_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    cout<<"------------------ Warpwise Strided Access ------------------"<<endl;
    for(int re = 1; re <= repeat; re++) {
        for(int i = 0; i < MEM_EXPR_TIME; i++) {
            status |= clSetKernelArg(mul_mix_kernel, 2, sizeof(int), &re);
            checkErr(status, ERR_SET_ARGUMENTS);
            status = clEnqueueNDRangeKernel(info.currentQueue, mul_mix_kernel, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);
        
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime = clEventTime(event);

            //throw away the first result
            if (i != 0)     mix_time[re] += tempTime;
        }  
        // status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*length, h_dest_values, 0, 0, 0);
        // checkErr(status, ERR_READ_BUFFER);

        // for(int i = 0; i < length; i++) {
        //     cout<<h_source_values_1[i]<<' '<<h_dest_values[i]<<endl;
        // }

    }

    for(int re = 1; re <= repeat; re++) {
        mix_time[re] /= (MEM_EXPR_TIME - 1);
        assert(mix_time[re] > 0);
        mix_throughput[re] = computeMem(localSize*gridSize*(re)*2, sizeof(T), mix_time[re]);
        cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re)
            <<"("<<localSize*gridSize*(re)<<')'
            <<" Time: "<<mix_time[re]<<" ms\t"<<"Throughput: "<<mix_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    //memory written back for not being optimized out
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);          

    //for python formating
    cout<<endl;
    cout<<"strided_bandwidth = ["<<strided_throughput[1];
    for(int re = 2; re <= repeat; re++) {      
        cout<<','<<strided_throughput[re];
    }               
    cout<<"]"<<endl;

    cout<<"coalesced_bandwidth = ["<<coalesced_throughput[1];
    for(int re = 2; re <= repeat; re++) {      
        cout<<','<<coalesced_throughput[re];
    }               
    cout<<"]"<<endl;

    cout<<"mix_bandwidth = ["<<mix_throughput[1];
    for(int re = 2; re <= repeat; re++) {      
        cout<<','<<mix_throughput[re];
    }               
    cout<<"]"<<endl;

    status = clReleaseMemObject(d_source_values);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    delete [] h_dest_values;

    delete[] coalesced_time;
    delete[] coalesced_throughput;
    delete[] strided_time;
    delete[] strided_throughput;
    delete[] mix_time;
    delete[] mix_throughput;

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
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/barrierKernel.cl");
    std::string kerAddr = path;

    char kerName_in[100] = "barrier_in";
    char kerName_free[100] = "barrier_free";

    KernelProcessor reader(&kerAddr,1,info.context);
    
    cl_kernel kernel_in = reader.getKernel(kerName_in);
    cl_kernel kernel_free = reader.getKernel(kerName_free);
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

void testAtomic(PlatInfo& info , double& totalTime, int localSize, int gridSize, bool isLocal) {
           
    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/atomicKernel.cl");
    std::string kerAddr = path;
    KernelProcessor reader(&kerAddr,1,info.context);

    const int repeatTime = ATOMIC_REPEAT_TIME;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)localSize * gridSize};

    if (isLocal) {      //test local atomic
        // int *testValueLocal = new int[gridSize];
        int testValueLocal = 0;
        // memset(testValueLocal, 0, sizeof(int) * gridSize);

        cl_mem d_source_local = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(int), NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(info.currentQueue, d_source_local, CL_TRUE, 0, sizeof(int), &testValueLocal, 0, 0, 0);

        char kerName_local[100] = "atomic_local";
        cl_kernel kernel_local = reader.getKernel(kerName_local);

        status |= clSetKernelArg(kernel_local, argsNum++, sizeof(cl_mem), &d_source_local);
        status |= clSetKernelArg(kernel_local, argsNum++, sizeof(int), &repeatTime);
        checkErr(status, ERR_SET_ARGUMENTS);

        double totalTime_local = 0.0;

        cl_event event;
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);

        for(int i = 0; i < ATOMIC_EXPR_TIME; i++) {
            status = clEnqueueNDRangeKernel(info.currentQueue, kernel_local, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);
            
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime_local = clEventTime(event);

            //throw away the first result
            if (i != 0)     totalTime_local += tempTime_local;
        }    
        totalTime_local /= (ATOMIC_EXPR_TIME - 1);

        status = clEnqueueReadBuffer(info.currentQueue, d_source_local, CL_TRUE, 0, sizeof(int), &testValueLocal, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);

        //checking
        cout<<"blockSize="<<localSize<<'\t'
            <<"gridSize="<<gridSize<<'\t'
            <<"\tTime: "<<totalTime_local<<" ms. Pertime: "<<totalTime_local / localSize / gridSize / ATOMIC_REPEAT_TIME * 1e6 <<" ns. ";

        // bool res = true;
        // for(int i = 0; i < gridSize; i++) {
        //     if (testValueLocal[i] != localSize * ATOMIC_REPEAT_TIME) {
        //         res = false;
        //         break;
        //     }
        // }
        // if (res)
        //     cout<<"local right!"<<endl;
        // else {
        //     cout<<"local wrong!"<<endl;
        // }
        if (testValueLocal == localSize * gridSize * ATOMIC_EXPR_TIME * ATOMIC_REPEAT_TIME) {
            cout<<"local right!"<<endl;
        }
        else {
            cout<<"local wrong!"<<endl;
            cout<<testValueLocal<<' '<<localSize * gridSize * ATOMIC_EXPR_TIME * ATOMIC_REPEAT_TIME<<endl;
        }
        status = clFinish(info.currentQueue);
        status = clReleaseMemObject(d_source_local);
        checkErr(status, ERR_RELEASE_MEM);

        // delete[] testValueLocal;

    }
    else {              //test global atomic
        int testValueGlobal = 0;
        
        cl_mem d_source_global = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(int), NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(info.currentQueue, d_source_global, CL_TRUE, 0, sizeof(int), &testValueGlobal, 0, 0, 0);


        char kerName_global[100] = "atomic_global";
        cl_kernel kernel_global = reader.getKernel(kerName_global);

        status |= clSetKernelArg(kernel_global, argsNum++, sizeof(cl_mem), &d_source_global);
        status |= clSetKernelArg(kernel_global, argsNum++, sizeof(int), &repeatTime);
        checkErr(status, ERR_SET_ARGUMENTS);

        double totalTime_global = 0.0;

        cl_event event;
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);

        for(int i = 0; i < ATOMIC_EXPR_TIME; i++) {

            status = clEnqueueNDRangeKernel(info.currentQueue, kernel_global, 1, 0, global, local, 0, 0, &event);
            clFlush(info.currentQueue);
            status = clFinish(info.currentQueue);
            
            checkErr(status, ERR_EXEC_KERNEL);
            double tempTime_global = clEventTime(event);

            //throw away the first result
            if (i != 0)     totalTime_global += tempTime_global;
        }    
        totalTime_global /= (ATOMIC_EXPR_TIME - 1);

        status = clEnqueueReadBuffer(info.currentQueue, d_source_global, CL_TRUE, 0, sizeof(int), &testValueGlobal, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);

        cout<<"blockSize="<<localSize<<'\t'
            <<"gridSize="<<gridSize<<'\t'
            <<"\tTime: "<<totalTime_global<<" ms. Pertime: "<<totalTime_global / localSize/gridSize/ATOMIC_REPEAT_TIME * 1e6 <<" ns. ";

        if (testValueGlobal == localSize * gridSize * ATOMIC_EXPR_TIME * ATOMIC_REPEAT_TIME)  {
                cout<<"Global right!"<<endl;
        } 
        else {
            cout<<"Global wrong!"<<endl;
        }
        status = clFinish(info.currentQueue);
        status = clReleaseMemObject(d_source_global);
        checkErr(status, ERR_RELEASE_MEM);
    }
}

typedef unsigned long ptr_type;

//test cache and memory latency for one workitem
void testLatency(PlatInfo& info) {

    int numOfSizes = 9, numOfStrides = 18;
    int testSize[9] = {16,32,64,128,256,512,1024,2048,4096};   //KB
    int strides[18] = {8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072, 262144, 524288, 1048576};   //Byte

    double latencyTime[9][19] = {0.0};

    //kernel reading
    char path[100] = PROJECT_ROOT;
    strcat(path, "/Kernels/latencyKernel.cl");
    std::string kerAddr = path;

    char kerName[100] = "latency";
    char add_address_kerName[100] = "add_address";

    KernelProcessor reader(&kerAddr,1,info.context);
    cl_kernel kernel = reader.getKernel(kerName);
    cl_kernel address_kernel = reader.getKernel(add_address_kerName);

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

template void testVPU<int>(int *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);
template void testVPU<double>(double *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

template void testMem<int>(PlatInfo& info , const int blockSize, const int gridSize, double& readTime, double& writeTime, double& mulTime,double& addTime, int repeat);
template void testMem<double>(PlatInfo& info ,  const int blockSize, const int gridSize,double& readTime, double& writeTime, double& mulTime, double& addTime, int repeat);

template void testAccess<int>(PlatInfo& info , const int blockSize, const int gridSize, int repeat);
template void testAccess<double>(PlatInfo& info ,  const int blockSize, const int gridSize, int repeat);