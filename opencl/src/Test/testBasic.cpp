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
void testMem(PlatInfo& info , const int localSize, const int gridSize, double& readTime, double& writeTime, double& mulTime, double& triadTime, int repeat) {

    bool res = true;
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

    T *h_source_values_1 = new T[length];  
    T *h_source_values_2 = new T[length];  
    T *h_dest_values = new T[length];

    for(int i = 0; i < length; i++) {
        h_source_values_1[i] = (T)i;
     	h_source_values_2[i] = (T)(i+10);
    }

    readTime = 0.0; writeTime = 0.0; mulTime = 0.0; triadTime = 0.0;

    //memory allocation
    cl_mem d_source_values_1 = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_1, CL_TRUE, 0, sizeof(T)*length, h_source_values_1, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    cl_mem d_source_values_2 = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_2, CL_TRUE, 0, sizeof(T)*length, h_source_values_2, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(T)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    char extra[500];
    if (sizeof(T) == sizeof(float))         strcpy(extra, "-DTYPE=float");
    else if (sizeof(T) == sizeof(double))   strcpy(extra, "-DTYPE=double");

    //kernel reading
    char path[100] = PROJECT_ROOT;
    char basicSizeName[20] = "";
    strcat(path, "/Kernels/memKernel.cl");
    std::string kerAddr = path;

    char read_kerName[100] = "mem_read";
    char write_kerName[100] = "mem_write";

    char mul_kerName[100] = "mem_mul";
    char triad_kerName[100] = "mem_triad";

    char mul_strided_kerName[100] = "mem_mul_strided";
    char mul_coalesced_kerName[100] = "mem_mul_coalesced";

    //get the kernel
    KernelProcessor reader(&kerAddr,1,info.context, extra);
    cl_kernel read_kernel = reader.getKernel(read_kerName);
    cl_kernel mul_kernel = reader.getKernel(mul_kerName);
    cl_kernel write_kernel = reader.getKernel(write_kerName);
    cl_kernel triad_kernel = reader.getKernel(triad_kerName);

    cl_kernel mul_strided_kernel = reader.getKernel(mul_strided_kerName);
     cl_kernel mul_coalesced_kernel = reader.getKernel(mul_coalesced_kerName);

    //set kernel arguments: read_kernel
    argsNum = 0;
    status |= clSetKernelArg(read_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(read_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

	//set kernel arguments: write_kernel
    argsNum = 0;
    status |= clSetKernelArg(write_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: triad_kernel
    argsNum = 0;
    status |= clSetKernelArg(triad_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(triad_kernel, argsNum++, sizeof(cl_mem), &d_source_values_2);
    status |= clSetKernelArg(triad_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_coalesced_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(mul_coalesced_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
    checkErr(status, ERR_SET_ARGUMENTS);

    //set kernel arguments: mul_strided_kernel
    argsNum = 0;
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_source_values_1);
    status |= clSetKernelArg(mul_strided_kernel, argsNum++, sizeof(cl_mem), &d_dest_values);
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
    
    //executing read, write, mul, triad
    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, triad_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     triadTime += tempTime;
    }    

    for(int i = 0; i < MEM_EXPR_TIME; i++) {

        status = clEnqueueNDRangeKernel(info.currentQueue, read_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     readTime += tempTime;
    }   

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, write_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     writeTime += tempTime;
    }    

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, mul_kernel, 1, 0, global, local, 0, 0, &event);
        clFlush(info.currentQueue);
        status = clFinish(info.currentQueue);
        
        checkErr(status, ERR_EXEC_KERNEL);
        double tempTime = clEventTime(event);

        //throw away the first result
        if (i != 0)     mulTime += tempTime;
    }    

    readTime /= ((MEM_EXPR_TIME - 1)*repeat);
    writeTime /= (MEM_EXPR_TIME - 1);
    mulTime /= (MEM_EXPR_TIME - 1);
    triadTime /= (MEM_EXPR_TIME - 1);


    //executing mul with coalesced and strided manner
    double *coalesced_time = new double[repeat];
    double *coalesced_throughput = new double[repeat];
	double *strided_time = new double[repeat];
    double *strided_throughput = new double[repeat];

    for(int i = 0; i < repeat; i++) {
        coalesced_time[i] = 0.0;
        coalesced_throughput[i] = 0.0;
        strided_time[i] = 0.0;
        strided_throughput[i] = 0.0;
    }

    //coalesced
    cout<<"------------------ Coalesced Access ------------------"<<endl;
    for(int re = 0; re < repeat; re++) {
	    for(int i = 0; i < MEM_EXPR_TIME; i++) {
	    	int repeatTime = re + 1;
	    	status |= clSetKernelArg(mul_coalesced_kernel, 2, sizeof(int), &repeatTime);
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

    for(int re = 0; re < repeat; re++) {
		coalesced_time[re] /= (MEM_EXPR_TIME - 1);
		assert(coalesced_time[re] > 0);
		coalesced_throughput[re] = computeMem(localSize*gridSize*(re+1)*2, sizeof(T), coalesced_time[re]);
    	cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re+1)
    		<<"("<<localSize*gridSize*(re+1)<<')'
    		<<" Time: "<<coalesced_time[re]<<" ms\t"<<"Throughput: "<<coalesced_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;

    //strided
    cout<<"------------------ Strided Access ------------------"<<endl;
    for(int re = 0; re < repeat; re++) {
	    for(int i = 0; i < MEM_EXPR_TIME; i++) {
	    	int repeatTime = re + 1;
	    	status |= clSetKernelArg(mul_strided_kernel, 2, sizeof(int), &repeatTime);
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

    for(int re = 0; re < repeat; re++) {
		strided_time[re] /= (MEM_EXPR_TIME - 1);
		assert(strided_time[re] > 0);
		strided_throughput[re] = computeMem(localSize*gridSize*(re+1)*2, sizeof(T), strided_time[re]);
    	cout<<"Data size: "<<localSize<<'*'<<gridSize<<'*'<<(re+1)
    		<<"("<<localSize*gridSize*(re+1)<<')'
    		<<" Time: "<<strided_time[re]<<" ms\t"<<"Throughput: "<<strided_throughput[re]<<" GB/s"<<endl;
    }
    cout<<endl;


    //memory written back for not being optimized out
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(T)*localSize*gridSize, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);                                        

    status = clReleaseMemObject(d_source_values_1);
    status = clReleaseMemObject(d_source_values_2);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values_1;
    delete [] h_source_values_2;
    delete [] h_dest_values;

    delete[] coalesced_time;
    delete[] coalesced_throughput;
    delete[] strided_time;
    delete[] strided_throughput;

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

template void testMem<float>(PlatInfo& info , const int blockSize, const int gridSize, double& readTime, double& writeTime, double& addTime, double& triadTime, int repeat);
template void testMem<double>(PlatInfo& info ,  const int blockSize, const int gridSize,double& readTime, double& writeTime, double& addTime, double& triadTime, int repeat);