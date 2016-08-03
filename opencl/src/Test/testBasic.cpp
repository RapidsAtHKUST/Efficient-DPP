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
#define FACTOR      (1.89)
#define REPEAT_TIME (60)

void testVPU(
    float *fixedValues, 
    int length, PlatInfo info , double& totalTime, int localSize, int gridSize, int basicSize) {
    
    if (basicSize != 1 && basicSize != 2 &&
        basicSize != 3 && basicSize != 4 &&
        basicSize != 8 && basicSize != 16) {
        std::cerr<<"Wrong parameter for basicSize."<<std::endl;
        exit(1);
    }

    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
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
    localSize, gridSize, info, FACTOR, REPEAT_TIME, basicSize);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(float)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    gettimeofday(&end, NULL);

    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    std::cout<<"basicSize: "<<basicSize<<endl;
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
}