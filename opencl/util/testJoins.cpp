////
////  TestJoins.cpp
////  gpuqp_opencl
////
////  Created by Zhuohang Lai on 5/21/15.
////  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
////
//
//#include "Plat.h"
//using namespace std;
//
//void recordRandom(int *a_keys, int *a_values, int length, int max) {
//    srand((unsigned)time(NULL));
//    sleep(1);
//    for(int i = 0; i < length ; i++) {
//        a_keys[i] = rand() % max;
//        a_values[i] = i;
//    }
//}
//
//void recordSorted_Only(Record* a, int len) {}
//void recordSorted(Record* a, int len) {}
//
//bool testHj(int rLen, int sLen) {
//
//    device_param_t param = Plat::get_device_param();
//
//    cl_int status = 0;
//    bool res = true;
//    FUNC_BEGIN;
//
//    float rSize, sSize;
//    rSize = rLen* 1.0*sizeof(int)/1024 * 2;     //key-value
//    if (rSize > 1024) {
//        rSize /= 1024;
//        std::cout<<"Table R tuples: "<<rLen<<" ("<<rSize<<" MB)"<<std::endl;
//    }
//    else {
//        std::cout<<"Table R tuples: "<<rLen<<" ("<<rSize<<" KB)"<<std::endl;
//    }
//
//    sSize = sLen* 1.0*sizeof(int)/1024 * 2;     //key-value
//    if (sSize > 1024) {
//        sSize /= 1024;
//        std::cout<<"Table S tuples: "<<sLen<<" ("<<sSize<<" MB)"<<std::endl;
//    }
//    else {
//        std::cout<<"Table S tuples: "<<sLen<<" ("<<sSize<<" KB)"<<std::endl;
//    }
//
//    double joinTime = 0, totalTime = 0;
//    //------------------------ test 1: No duplicate key values ------------------------
//
////    cout<<"--- Test 1 : no duplicate---"<<endl;
//    int *h_R_keys = new int[rLen];
//    int *h_R_values= new int[rLen];
//
//    int *h_S_keys = new int[sLen];
//    int *h_S_values = new int[sLen];
//
//    int *h_Out = NULL;
//
//    recordRandom(h_R_keys, h_R_values, rLen, rLen);
//    recordRandom(h_S_keys, h_S_values, sLen, rLen);
//
//    struct timeval start, end;
//
//    gettimeofday(&start,NULL);
//
//    //memory allocation
//    cl_mem d_R_keys = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*rLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//    cl_mem d_R_values = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*rLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//
//    cl_mem d_S_keys = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*sLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//    cl_mem d_S_values = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*sLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//
//
//    status = clEnqueueWriteBuffer(param.queue, d_R_keys, CL_TRUE, 0, sizeof(int)*rLen, h_R_keys, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//    status = clEnqueueWriteBuffer(param.queue, d_R_values, CL_TRUE, 0, sizeof(int)*rLen, h_R_values, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//
//    status = clEnqueueWriteBuffer(param.queue, d_S_keys, CL_TRUE, 0, sizeof(int)*sLen, h_S_keys, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//    status = clEnqueueWriteBuffer(param.queue, d_S_values, CL_TRUE, 0, sizeof(int)*sLen, h_S_values, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//
//
//    //call gather
//    int d_res_len;
//    joinTime = hashjoin_np(d_R_keys, d_R_values, rLen, d_S_keys, d_S_values, sLen, d_res_len);
//    gettimeofday(&end, NULL);
//
//    totalTime = diffTime(end, start);
//
//    std::cout<<"---- Kernel execution ended ----"<<std::endl;
////    int h_res_len= 0;
////    for(int r = 0; r < rLen; r++) {
////        for(int s = 0; s < sLen; s++) {
////            if (h_R_keys[r] == h_S_keys[s])   h_res_len++;
////        }
////    }
////
////    if (d_res_len == h_res_len) {
////        std::cout<<"Hash join test passes!"<<std::endl;
////        std::cout<<"# joined results:"<<d_res_len<<std::endl;
////    }
////    else {
////        std::cout<<"Hash join test fails!"<<std::endl;
////        std::cout<<"CPU:"<<h_res_len<<'\t'<<"GPU:"<<d_res_len<<std::endl;
////    }
//    //suppose the result is correct!
//    cout<<"Joined results:"<<d_res_len<<endl;
//    cout<<"Join time: "<<joinTime<<" ms."<<endl;
//
//    float throughput = (rLen+sLen)* 1.0*sizeof(int)*2/1024/1024/1024 / totalTime * 1000;
//    cout<<"Total Execution time: "<<totalTime<<" ms ("<<throughput<<" GB/s)"<<endl;
////    if (oLen != 0) {
////        h_Out = new Record[oLen];
////
////        //memory written back
////        status = clEnqueueReadBuffer(param.queue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
////        checkErr(status, ERR_READ_BUFFER);
////        status = clFinish(param.queue);
////    }
//
////
////    SHOW_CHECKING;
////    if (oLen != 0) {
////        //check
////        for(int i = 0 ; i < oLen; i++) {
////            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
////                res = false;
////                break;
////            }
////        }
////        delete [] h_Out;
////        status = clReleaseMemObject(d_Out);
////        checkErr(status, ERR_RELEASE_MEM);
////    }
////    int smallRes = rLen;
////    if (smallRes > sLen)    smallRes = sLen;
////    if (oLen != smallRes)   res = false;
//
//    status = clReleaseMemObject(d_R_keys);
//    status = clReleaseMemObject(d_R_values);
//
//    checkErr(status,ERR_RELEASE_MEM);
//    status = clReleaseMemObject(d_S_keys);
//    status = clReleaseMemObject(d_S_values);
//
//    checkErr(status,ERR_RELEASE_MEM);
//    delete [] h_R_keys;
//    delete [] h_R_values;
//
//    delete [] h_S_values;
//    delete [] h_S_keys;
//
////    FUNC_CHECK(res);
////    SHOW_TIME(totalTime);
////    SHOW_TOTAL_TIME(diffTime(end, start));
////    cout<<"--- End of Test 1 ---"<<endl<<endl;
//
//    //---------------------------------- End fo test 1 -----------------------------------
//
//    //------------------------ test 2: Having duplicate key values ------------------------
//
////    cout<<"--- Test 2 : has duplicate---"<<endl;
////
////    res = true;
////    h_R = new Record[rLen];
////    h_S = new Record[sLen];
////
////    recordRandom(h_R, rLen);
////    recordRandom(h_S, sLen);
////
////    gettimeofday(&start,NULL);
////
////    //memory allocation
////    d_R = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
////    checkErr(status, ERR_HOST_ALLOCATION);
////    d_S = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
////    checkErr(status, ERR_HOST_ALLOCATION);
////
////    status = clEnqueueWriteBuffer(param.queue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
////    checkErr(status, ERR_WRITE_BUFFER);
////    status = clEnqueueWriteBuffer(param.queue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
////    checkErr(status, ERR_WRITE_BUFFER);
////
////    //partition
////    partitionHJ(d_R, rLen, countBit, info, localSize, gridSize);
////    partitionHJ(d_S, sLen, countBit, info, localSize, gridSize);
////
////    //call gather
////    double totalTime2 = hj(d_R, rLen, d_S, sLen, d_Out, oLen, info, countBit, localSize);
////
////    if (oLen != 0) {
////        h_Out = new Record[oLen];
////
////        //memory written back
////        status = clEnqueueReadBuffer(param.queue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
////        checkErr(status, ERR_READ_BUFFER);
////        status = clFinish(param.queue);
////    }
////    gettimeofday(&end, NULL);
////
////    SHOW_CHECKING;
////    if (oLen != 0) {
////        //check
////        for(int i = 0 ; i < oLen; i++) {
////            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
////                res = false;
////                break;
////            }
////        }
////        delete [] h_Out;
////        status = clReleaseMemObject(d_Out);
////        checkErr(status, ERR_RELEASE_MEM);
////    }
////
////    totalTime = (totalTime1 + totalTime2) / 2;
////
////    status = clReleaseMemObject(d_R);
////    checkErr(status,ERR_RELEASE_MEM);
////    status = clReleaseMemObject(d_S);
////    checkErr(status,ERR_RELEASE_MEM);
////    delete [] h_R;
////    delete [] h_S;
////
////    FUNC_CHECK(res);
////    SHOW_JOIN_RESULT(oLen);
////    SHOW_TIME(totalTime2);
////    SHOW_TOTAL_TIME(diffTime(end, start));
////    cout<<"--- End of Test 2 ---"<<endl<<endl;
////
////    //---------------------------------- End fo test 2 -----------------------------------
////    FUNC_END;
//
//    return res;
//}
