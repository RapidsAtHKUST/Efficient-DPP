//
//  TestPrimitives.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/21/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "testPrimitives.h"
#include "Foundation.h"
#include "gpuqpHeaders.h"
using namespace std;

bool testMap(Record *fixedSource, int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    Record *h_source = new Record[length];
    Record *h_dest = new Record[length];
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source[i].x = fixedSource[i].x;
        h_source[i].y = fixedSource[i].y;
    }

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //call map
    totalTime = map(d_source, length, d_dest, localSize, gridSize, info);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest, CL_TRUE, 0, sizeof(Record)*length, h_dest, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);

    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if (h_source[i].x != h_dest[i].x || floorOfPower2(h_source[i].y) != h_dest[i].y) res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest);
    checkErr(status, ERR_RELEASE_MEM);

    
    delete [] h_source;
    delete [] h_dest;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testGather(Record *fixedSource, int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    Record *h_source = new Record[length];
    Record *h_dest = new Record[length];
    int *h_loc = new int[length];
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source[i].x = fixedSource[i].x;
        h_source[i].y = fixedSource[i].y;
    }
    
    intRandom_Only(h_loc, length, length);
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*length, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    totalTime = gather(d_source, d_dest, length, d_loc, localSize, gridSize, info);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest, CL_TRUE, 0, sizeof(Record)*length, h_dest, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if ( (h_dest[i].x != h_source[h_loc[i]].x ) ||
             (h_dest[i].y != h_source[h_loc[i]].y ) )    res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
    
    delete [] h_source;
    delete [] h_dest;
    delete [] h_loc;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testScatter(Record *fixedSource, int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    Record *h_source = new Record[length];
    Record *h_dest = new Record[length];
    int *h_loc = new int[length];
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source[i].x = fixedSource[i].x;
        h_source[i].y = fixedSource[i].y;
    }
    
    intRandom_Only(h_loc, length, length);
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*length, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    totalTime = scatter(d_source, d_dest, length, d_loc,localSize,gridSize,info);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest, CL_TRUE, 0, sizeof(Record)*length, h_dest, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if ( (h_dest[h_loc[i]].x != h_source[i].x ) ||
            (h_dest[h_loc[i]].y != h_source[i].y ) )    res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
    
    delete [] h_source;
    delete [] h_dest;
    delete [] h_loc;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testScan(int *fixedSource, int length, PlatInfo info, double& totalTime, int isExclusive, int localSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, "not fixed");
    SHOW_DATA_NUM(length);
    
    if (isExclusive == 0)   cout<<"Type: Inclusive."<<endl;
    else                    cout<<"Type: Exclusive."<<endl;
    
    int *gpu_io = new int[length];
    int *cpu_input = new int[length];
    int *cpu_output = new int[length];
    
    intRandom(gpu_io, length, INT_MAX);
    
    for(int i = 0; i < length; i++) {
        cpu_input[i] = gpu_io[i];
    }
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    cl_mem cl_arr = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)* length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    //write buffer to the cl_mem
    status = clEnqueueWriteBuffer(info.currentQueue, cl_arr, CL_TRUE, 0, sizeof(int)*length, gpu_io, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    totalTime = scan(cl_arr, length, isExclusive,info, localSize);
    
    status = clEnqueueReadBuffer(info.currentQueue, cl_arr, CL_TRUE, 0, sizeof(int)*length, gpu_io, 0, NULL, NULL);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    if (isExclusive == 0) {         //inclusive
        cpu_output[0] = cpu_input[0];
        for(int i = 1 ; i < length; i++) {
            cpu_output[i] = cpu_input[i] + cpu_output[i-1];
        }
    }
    else {                          //exclusive
        cpu_output[0] = 0;
        for(int i = 1 ; i < length; i++) {
            cpu_output[i] = cpu_output[i-1] + cpu_input[i-1];
        }
    }
    
    for(int i = 0; i < length; i++) {
        if (cpu_output[i] != gpu_io[i]) res = false;
    }
    FUNC_CHECK(res);
    
    //release
    status = clReleaseMemObject(cl_arr);
    checkErr(status, ERR_RELEASE_MEM);
    
    delete [] gpu_io;
    delete [] cpu_input;
    delete [] cpu_output;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testSplit(Record *fixedSource, int length, PlatInfo info , int fanout, double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    cout<<"fanout: "<<fanout<<endl;
    
    Record *h_source = new Record[length];
    Record *h_dest = new Record[length];
    
    recordRandom(h_source, length, fanout);
//    for(int i = 0; i < length; i++) {
//        h_source[i].x = fixedSource[i].x;
//        h_source[i].y = fixedSource[i].y;
//    }
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    totalTime = split(d_source, d_dest, length, fanout, info, localSize, gridSize);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest, CL_TRUE, 0, sizeof(Record)*length, h_dest, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    for(int i = 1; i < length; i++) {
        if (h_dest[i].y < h_dest[i-1].y)  res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
    
    delete [] h_source;
    delete [] h_dest;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testRadixSort(Record *fixedSource, int length, PlatInfo info, double& totalTime) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(256, 512);
    SHOW_DATA_NUM(length);
    
    Record *h_source = new Record[length];
    Record *cpuInput = new Record[length];
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source[i].x = fixedSource[i].x;
        h_source[i].y = fixedSource[i].y;
    }
    
    
    for(int i = 0;i<length;i++) cpuInput[i] = h_source[i];
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather

    totalTime = radixSort(d_source, length, info);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    qsort(cpuInput, length, sizeof(Record), compRecordAsc);
    
    for(int i = 0;i<length;i++)    {
        if (h_source[i].y != cpuInput[i].y) {
            res = false;
            break;
        }
    }
    
    delete [] h_source;
    delete [] cpuInput;
    
    status = clReleaseMemObject(d_source);
    checkErr(status,ERR_RELEASE_MEM);
    
    FUNC_CHECK(res);
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testBitonitSort(Record *fixedSource, int length, PlatInfo info, int dir, double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize , gridSize);
    SHOW_DATA_NUM(length);
    if (dir == 0)   cout<<"Direction: Descending."<<endl;
    else            cout<<"Direction: Ascending."<<endl;
    
    Record *h_source = new Record[length];
    Record *cpuInput = new Record[length];
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source[i].x = fixedSource[i].x;
        h_source[i].y = fixedSource[i].y;
    }
    
    for(int i = 0;i<length;i++) cpuInput[i] = h_source[i];
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    totalTime = bisort(d_source, length, dir, info, localSize, gridSize);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source, CL_TRUE, 0, sizeof(Record)*length, h_source, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    if (dir == 0) qsort(cpuInput, length, sizeof(Record), compRecordDec);
    else          qsort(cpuInput, length, sizeof(Record), compRecordAsc);
    
    for(int i = 0;i<length;i++)    {
        if (h_source[i].y != cpuInput[i].y) {
            res = false;
            break;
        }
    }
    
    delete [] h_source;
    delete [] cpuInput;
    
    status = clReleaseMemObject(d_source);
    checkErr(status,ERR_RELEASE_MEM);
    
    FUNC_CHECK(res);
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}


