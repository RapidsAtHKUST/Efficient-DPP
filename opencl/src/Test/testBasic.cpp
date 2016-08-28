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
    if (sizeof(T) == sizeof(float))         
        strcpy(extra, "-DTYPE=float -DTYPE2=float2 -DTYPE4=float4 -DTYPE8=float8 -DTYPE16=float16");
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
void testMemReadWrite(
    T *fixedValues, PlatInfo& info , double& readTime, double& writeTime, int localSize, int gridSize, int basicSize) {

    bool res = true;
    int length = localSize * gridSize * basicSize ;
    int vec_length = localSize * gridSize ;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif
    T *h_source_values = new T[length];  
    T *h_dest_values = new T[localSize * gridSize];

    for(int i = 0; i < length; i++) {
        h_source_values[i] = fixedValues[i];
    }
    
    //memory allocation
    cl_int status = 0;
    readTime = 0;
    writeTime = 0;

    int argsNum = 0;
    
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(T)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    //for each thread to write out so that the code would not be optimized
    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(T)*localSize*gridSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    char extra[500];
    if (sizeof(T) == sizeof(float))         
        strcpy(extra, "-DTYPE=float -DTYPE2=float2 -DTYPE4=float4 -DTYPE8=float8 -DTYPE16=float16");
    else if (sizeof(T) == sizeof(double))
        strcpy(extra, "-DTYPE=double -DTYPE2=double2 -DTYPE4=double4 -DTYPE8=double8 -DTYPE16=double16");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;
    my_itoa(basicSize, basicSizeName, 10);

    char read_write_kerName[100] = "mem_read_write";
    char write_kerName[100] = "mem_write";

    strcat(read_write_kerName, basicSizeName);
    strcat(write_kerName, basicSizeName);

    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel read_write_kernel = reader.getKernel(read_write_kerName);
    cl_kernel write_kernel = reader.getKernel(write_kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(read_write_kernel, argsNum++, sizeof(cl_mem), &d_source_values);
    status |= clSetKernelArg(read_write_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    argsNum = 0;
    status |= clSetKernelArg(write_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
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

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, write_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     writeTime += tempTime;

        status = clEnqueueNDRangeKernel(info.currentQueue, read_write_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     readTime += tempTime;
    }    
    writeTime /= (MEM_EXPR_TIME - 1);
    readTime /= (MEM_EXPR_TIME - 1);
    readTime -= writeTime;

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*localSize*gridSize, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);                                        

    status = clReleaseMemObject(d_source_values);
    status = clReleaseMemObject(d_dest_values);

    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    delete [] h_dest_values;

#ifndef SILENCE
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
#endif
}

template<typename T>
void testTriad(T* fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize) {

    bool res = true;
    int length = localSize * gridSize * basicSize;
    int vec_length = localSize * gridSize;
#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    T *h_dest_values_a = new T[length];  
    T *h_source_values_b = new T[length];  
    T *h_source_values_c = new T[length];  

    for(int i = 0; i < length ;i++) {
        h_source_values_b[i] = fixedValues[i];
        h_source_values_c[i] = fixedValues[i] + i;
    }

    //memory allocation
    cl_int status = 0;
    totalTime = 0;
    int argsNum = 0;
    
    cl_mem d_source_values_b = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_source_values_c = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest_values_a = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_b, CL_TRUE, 0, sizeof(T)*length, h_source_values_b, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);  

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_c, CL_TRUE, 0, sizeof(T)*length, h_source_values_c, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);  

    char extra[500];
    if (sizeof(T) == sizeof(float))         
        strcpy(extra, "-DTYPE=float -DTYPE2=float2 -DTYPE4=float4 -DTYPE8=float8 -DTYPE16=float16");
    else if (sizeof(T) == sizeof(double))
        strcpy(extra, "-DTYPE=double -DTYPE2=double2 -DTYPE4=double4 -DTYPE8=double8 -DTYPE16=double16");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;
    my_itoa(basicSize, basicSizeName, 10);
    char kerName[100] = "triad";
    strcat(kerName, basicSizeName);

    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel kernel = reader.getKernel(kerName);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_dest_values_a);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values_b);
    status |= clSetKernelArg(kernel, argsNum++, sizeof(cl_mem), &d_source_values_c);
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

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);
        //throw away the first result
        if (i != 0)     totalTime += tempTime;
    }    
    totalTime /= (MEM_EXPR_TIME - 1);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values_a, CL_TRUE, 0, sizeof(T)*length, h_dest_values_a, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);
    status = clReleaseMemObject(d_dest_values_a);
    status = clReleaseMemObject(d_source_values_b);
    status = clReleaseMemObject(d_source_values_c);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_dest_values_a;
    delete [] h_source_values_b;
    delete [] h_source_values_c;

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

template void testVPU<float>(float *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);
template void testVPU<double>(double *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

template void testMemReadWrite<float>(
    float *fixedValues, PlatInfo& info ,  double& readTime, double& writeTime, int localSize, int gridSize, int basicSize);
 template void testMemReadWrite<double>(
    double *fixedValues, PlatInfo& info , double& readTime, double& writeTime, int localSize, int gridSize, int basicSize);

template void testTriad<float>(float* fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);
template void testTriad<double>(double* fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

