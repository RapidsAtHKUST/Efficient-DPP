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

void recordRandom(Record *a, int b, int c) {}

bool testMap(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, 
    int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    int *h_source_values = new int[length];
    int *h_dest_values = new int[length];    
#ifdef RECORDS
    int *h_source_keys = new int[length];
    int *h_dest_keys = new int[length];    
#endif
    
//    recordRandom(h_source, length);
    for(int i = 0; i < length; i++) {
        h_source_values[i] = fixedValues[i];
#ifdef RECORDS
        h_source_keys[i] = fixedKeys[i];
#endif
    }

    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_int status = 0;
    
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#ifdef RECORDS
    cl_mem d_source_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest_keys = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_keys, CL_TRUE, 0, sizeof(int)*length, h_source_keys, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#endif    

    //call map
    totalTime = map(
#ifdef RECORDS
    d_source_keys, d_dest_keys,true,
#endif
    d_source_values, d_dest_values, length, 
    localSize, gridSize, info);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
#ifdef RECORDS
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_keys, CL_TRUE, 0, sizeof(int)*length, h_dest_keys, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
#endif
    status = clFinish(info.currentQueue);
    
    gettimeofday(&end, NULL);

    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if (
#ifdef RECORDS
        h_source_keys[i] != h_dest_keys[i]    ||
#endif
        // h_source_values[i] != h_dest_values[i])
        floorOfPower2_CPU(h_source_values[i]) != h_dest_values[i]) 
            res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    delete [] h_dest_values;
#ifdef RECORDS
    status = clReleaseMemObject(d_source_keys);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_keys);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_keys;
    delete [] h_dest_keys;
#endif

    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testGather(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    int *h_source_values = new int[length];
    int *h_dest_values = new int[length];
#ifdef RECORDS
    int *h_source_keys = new int[length];
    int *h_dest_keys = new int[length];    
#endif

    int *h_loc = new int[length];
    
    for(int i = 0; i < length; i++) {
#ifdef RECORDS
        h_source_keys[i] = fixedKeys[i];
#endif
        h_source_values[i] = fixedValues[i];        
    }
    
    valRandom_Only<int>(h_loc, length, length);
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
#ifdef RECORDS
    cl_mem d_source_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest_keys = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_keys, CL_TRUE, 0, sizeof(int)*length, h_source_keys, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#endif  

    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*length, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //call gather
    totalTime = gather(
#ifdef RECORDS
    d_source_keys, d_dest_keys,true,
#endif
    d_source_values, d_dest_values, length, d_loc, localSize, gridSize, info);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
#ifdef RECORDS
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_keys, CL_TRUE, 0, sizeof(int)*length, h_dest_keys, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
#endif
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if ( 
#ifdef RECORDS
            (h_dest_keys[i] != h_source_keys[h_loc[i]]) ||
#endif
             (h_dest_values[i] != h_source_values[h_loc[i]] ) )    res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
#ifdef RECORDS
    status = clReleaseMemObject(d_source_keys);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_keys);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_keys;
    delete [] h_dest_keys;
#endif

    delete [] h_source_values;
    delete [] h_dest_values;
    delete [] h_loc;
    
    SHOW_TIME(totalTime);
    SHOW_TOTAL_TIME(diffTime(end, start));
    FUNC_END;
    
    return res;
}

bool testScatter(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, int length, PlatInfo info , double& totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_DATA_NUM(length);
    
    int *h_source_values = new int[length];
    int *h_dest_values = new int[length];
#ifdef RECORDS
    int *h_source_keys = new int[length];
    int *h_dest_keys = new int[length];    
#endif

    int *h_loc = new int[length];
    
    for(int i = 0; i < length; i++) {
#ifdef RECORDS
        h_source_keys[i] = fixedKeys[i];
#endif
        h_source_values[i] = fixedValues[i];        
    }
    
    valRandom_Only<int>(h_loc, length, length);
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    // cl_int16 *int_16_source_values = (cl_int16 *)h_source_values; 
    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
#ifdef RECORDS
    cl_mem d_source_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_dest_keys = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_keys, CL_TRUE, 0, sizeof(int)*length, h_source_keys, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#endif  

    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*length, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //call gather
    totalTime = scatter(
#ifdef RECORDS
    d_source_keys, d_dest_keys,true,
#endif
    d_source_values, d_dest_values, length, d_loc, localSize, gridSize, info);
    
    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
    
#ifdef RECORDS
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_keys, CL_TRUE, 0, sizeof(int)*length, h_dest_keys, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
#endif
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    for(int i = 0; i < length; i++) {
        if ( 
#ifdef RECORDS
            (h_dest_keys[h_loc[i]] != h_source_keys[i]) ||
#endif
             (h_dest_values[h_loc[i]] != h_source_values[i] ) )    res = false;
    }
    FUNC_CHECK(res);
    
    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
#ifdef RECORDS
    status = clReleaseMemObject(d_source_keys);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_keys);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_keys;
    delete [] h_dest_keys;
#endif

    delete [] h_source_values;
    delete [] h_dest_values;
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
    
    // valRandom<int>(gpu_io, length, 1500);
    for(int i = 0; i < length; i++) gpu_io[i] = 1;
    
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
    
    totalTime = scan_ble(cl_arr, length, isExclusive,info, localSize);
    // totalTime = scan_blelloch(cl_arr, length, isExclusive,info, localSize);
    
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
        if (cpu_output[i] != gpu_io[i])  {
            res = false;
            // std::cout<<cpu_output[i-1]<<' '<<gpu_io[i-1]<<std::endl;
            
            // std::cout<<cpu_output[i]<<' '<<gpu_io[i]<<std::endl;

            break;
        }
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

bool testRadixSort(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, 
    int length, PlatInfo info, double& totalTime) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(256, 512);
    SHOW_DATA_NUM(length);

#ifdef RECORDS
    int *h_source_keys = new int[length];
#endif
    int *h_source_values = new int[length];
    int *cpu_input = new int[length];

    for(int i = 0; i < length; i++) {
#ifdef RECORDS
        h_source_keys[i] = fixedKeys[i];
#endif
        h_source_values[i] = fixedValues[i];        
        cpu_input[i] = fixedValues[i];
    }
    
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    cl_int status = 0;
    
    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
#ifdef RECORDS
    cl_mem d_source_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
#endif
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#ifdef RECORDS
    status = clEnqueueWriteBuffer(info.currentQueue, d_source_keys, CL_TRUE, 0, sizeof(int)*length, h_source_keys, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
#endif

    //call radix sort
    totalTime = radixSort(
#ifdef RECORDS
    d_source_keys, true,   
#endif
    d_source_values, length, info);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
#ifdef RECORDS
    status = clEnqueueReadBuffer(info.currentQueue, d_source_keys, CL_TRUE, 0, sizeof(int)*length, h_source_keys, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);
    status = clFinish(info.currentQueue);
#endif
    
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    sort(cpu_input, cpu_input + length);
    
    for(int i = 0;i<length;i++)    {
        if (h_source_values[i] != cpu_input[i]) {
            res = false;
            break;
        }
    }
    
    delete [] h_source_values;
#ifdef RECORDS
    delete [] h_source_keys;
#endif
    delete [] cpu_input;
    
    status = clReleaseMemObject(d_source_values);
    checkErr(status,ERR_RELEASE_MEM);
#ifdef RECORDS
    status = clReleaseMemObject(d_source_keys);
    checkErr(status,ERR_RELEASE_MEM);
#endif
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
    
//    cout<<"Output:"<<endl;
//    for(int i = 0; i < length; i++) {
//        cout<<h_source[i].x<<' '<<h_source[i].y<<endl;
//    }
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


