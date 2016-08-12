//
//  TestPrimitives.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/21/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
using namespace std;

//for instruction throughput VPU test
void testVPU(
    float *fixedValues, 
    int length, PlatInfo info , double& totalTime, int localSize, int gridSize, int basicSize) {
    
    if (basicSize != 1 && basicSize != 2 &&
        basicSize != 3 && basicSize != 4 &&
        basicSize != 8 && basicSize != 16) {
        cout<<"wrong basicSize: "<<basicSize<<endl;
        std::cerr<<"Wrong parameter for basicSize."<<std::endl;
        exit(1);
    }

    bool res = true;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    float *h_source_values = new float[length];  

    for(int i = 0; i < length; i++) {
        h_source_values[i] = fixedValues[i];
    }

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    //call vpu
    totalTime = vpu(
    d_source_values, length, 
    localSize, gridSize, info, FACTOR, VPU_REPEAT_TIME, basicSize);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    gettimeofday(&end, NULL);

    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;

#ifndef SILENCE
    std::cout<<"Variable type: "<<"float"<<basicSize<<endl;
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
#endif
}

//for memory read bandwidth test
void testMemRead(
    float *fixedValues, 
    int length, PlatInfo info , double& totalTime, int localSize, int gridSize, int basicSize) {

    bool res = true;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    float *h_source_values = new float[length];  
    float *h_dest_values = new float[localSize * gridSize];

    for(int i = 0; i < length; i++) {
        h_source_values[i] = fixedValues[i];
    }
    // cout<<endl;

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    //for each thread to write out so that the code would not be optimized
    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(float)*localSize*gridSize, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //call vpu
    totalTime = mem_read(
    d_source_values, d_dest_values, length, 
    localSize, gridSize, info, FACTOR, basicSize);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(float)*localSize*gridSize, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    gettimeofday(&end, NULL);


    //testing
    // float sum = 0.0;
    // for(int i = 0; i < localSize * gridSize; i++) {
        // cout<<localSize * gridSize<<':'<<h_dest_values[0]<<endl;
    
    // if (abs(sum - 1.7682*length) < 1e3)  cout<<"correct!"<<endl;
    // else      {
    //      cout<<"wrong!"<<sum-1.7682 * length<<endl;
    // }                                         


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

//for memory write bandwidth test
void testMemWrite(int length, PlatInfo info , double& totalTime, int localSize, int gridSize, int basicSize) {

    bool res = true;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    float *h_source_values = new float[length];  

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    //call vpu
    totalTime = mem_write(
    d_source_values, length, 
    localSize, gridSize, info, FACTOR, basicSize);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    gettimeofday(&end, NULL);

    status = clReleaseMemObject(d_source_values);

    checkErr(status, ERR_RELEASE_MEM);

    // for(int i = 439087; i < 439532; i++) {
    //     cout<<h_source_values[i]<<' ';
    //     // if (abs(h_source_values[i]-1.0875) > 1e-3) {
    //     //     cout<<i<<": "<<h_source_values[i]<<' ';
    //     //     cerr<<"wrong answer!"<<endl;
    //     //     exit(1);
    //     // }
    // }
    // cout<<endl;

    delete [] h_source_values;

#ifndef SILENCE
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
#endif
}

void testTriad(
    float *fixedValues, 
    int length, PlatInfo info , double& totalTime, int localSize, int gridSize, int basicSize) {
    
    if (basicSize != 1 && basicSize != 2 &&
        basicSize != 3 && basicSize != 4 &&
        basicSize != 8 && basicSize != 16) {
        cout<<"wrong basicSize: "<<basicSize<<endl;
        std::cerr<<"Wrong parameter for basicSize."<<std::endl;
        exit(1);
    }

    bool res = true;

#ifndef SILENCE
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
#endif

    float *h_source_values_b = new float[length];  
    float *h_source_values_c = new float[length];  
    float *h_dest_values_a = new float[length];  


    for(int i = 0; i < length; i++) {
        h_source_values_b[i] = fixedValues[i];
        h_source_values_c[i] = h_source_values_b[i] * 0.5 + i - 1.34;
    }

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source_values_b = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_source_values_c = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest_values_a = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_b, CL_TRUE, 0, sizeof(float)*length, h_source_values_b, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);   

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values_c, CL_TRUE, 0, sizeof(float)*length, h_source_values_c, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);  
    //call vpu
    totalTime = triad(
    d_source_values_b, d_source_values_c, d_dest_values_a, length, 
    localSize, gridSize, info, basicSize);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values_a, CL_TRUE, 0, sizeof(float)*length, h_dest_values_a, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    gettimeofday(&end, NULL);

    status = clReleaseMemObject(d_source_values_b);
    status = clReleaseMemObject(d_source_values_c);
    status = clReleaseMemObject(d_dest_values_a);

    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_dest_values_a;
    delete [] h_source_values_b;
    delete [] h_source_values_c;


#ifndef SILENCE
    std::cout<<"Variable type: "<<"float"<<basicSize<<endl;
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
#endif
}