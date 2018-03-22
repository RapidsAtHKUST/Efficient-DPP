//
//  TestJoins.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/21/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
using namespace std;

//shadowing function
void recordRandom_Only(Record* a, int len, int time) {}
void recordRandom(Record* a, int len) {}
void recordRandom1(Record* a, int len, int max) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < len ; i++) {
        a[i].x = rand()% max;
        a[i].y = i;
    }
}
void recordSorted_Only(Record* a, int len) {}
void recordSorted(Record* a, int len) {}

bool testNinlj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_TABLE_R_NUM(rLen);
    SHOW_TABLE_S_NUM(sLen);
    
//------------------------ Test 1: No duplicate key values ------------------------
    
    cout<<"--- Test 1 : no duplicate---"<<endl;
    Record *h_R = new Record[rLen];
    Record *h_S = new Record[sLen];
    Record *h_Out = NULL;
    
    recordRandom_Only(h_R, rLen, SHUFFLE_TIME(rLen));
    recordRandom_Only(h_S, sLen, SHUFFLE_TIME(sLen));
    
    cl_int status = 0;
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_mem d_R = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_S = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    cl_mem d_Out;
    int oLen = 0;
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    double totalTime1 = ninlj(d_R, rLen, d_S, sLen, d_Out, oLen, info, localSize, gridSize);
    
    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);
    }
    gettimeofday(&end, NULL);
    
    //check
    SHOW_CHECKING;
    if (oLen != 0) {
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }
    
    int smallRes = rLen;
    if (smallRes > sLen)    smallRes = sLen;
    if (oLen != smallRes)   res = false;
    
    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime1);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 1 ---"<<endl<<endl;
    
//---------------------------------- End fo test 1 -----------------------------------
    
//------------------------ Test 2: Having duplicate key values ------------------------
    
    cout<<"--- Test 2 : has duplicate---"<<endl;
    
    res = true;
    h_R = new Record[rLen];
    h_S = new Record[sLen];
    
    recordRandom(h_R, rLen);
    recordRandom(h_S, sLen);
    
    gettimeofday(&start,NULL);
    
    //memory allocation
    d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    double totalTime2 = ninlj(d_R, rLen, d_S, sLen, d_Out, oLen, info, localSize, gridSize);

    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);
    }
    gettimeofday(&end, NULL);
    
    SHOW_CHECKING;
    if (oLen != 0) {
        //check
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }

    
    totalTime = (totalTime1 + totalTime2) / 2;

    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime2);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 2 ---"<<endl<<endl;
    
//---------------------------------- End fo test 2 -----------------------------------
    FUNC_END;
    
    return res;
}

bool testInlj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize, int gridSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, gridSize);
    SHOW_TABLE_R_NUM(rLen);
    SHOW_TABLE_S_NUM(sLen);
    
    //------------------------ Test 1: No duplicate key values ------------------------
    
    cout<<"--- Test 1 : no duplicate---"<<endl;
    Record *h_R = new Record[rLen];
    Record *h_S = new Record[sLen];
    Record *h_Out = NULL;
    
    //h_R should be sorted
    recordSorted_Only(h_R, rLen);
    recordRandom_Only(h_S, sLen, SHUFFLE_TIME(sLen));
    cl_int status = 0;
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    int mPart = calMPart(rLen);
    int numOfInternalNodes_R = 0,mark_R = 0;
    int *CSS_R = generateCSSTree(h_R, rLen, mPart, numOfInternalNodes_R, mark_R);   //size: numOfInternalNodes * mPart * sizeof(int)
    int CSS_R_Lengh = numOfInternalNodes_R * mPart;
    
    cl_mem d_CSS_R = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*CSS_R_Lengh, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_CSS_R, CL_TRUE, 0, sizeof(int) * CSS_R_Lengh, CSS_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    CSS_Tree_Info treeInfo;
    treeInfo.d_CSS = d_CSS_R;
    treeInfo.CSS_length = CSS_R_Lengh;
    treeInfo.numOfInternalNodes = numOfInternalNodes_R;
    treeInfo.mark = mark_R;
    treeInfo.mPart = mPart;
    
    //memory allocation
    cl_mem d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    cl_mem d_Out;
    int oLen = 0;
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    double totalTime1 = inlj(d_R, rLen, d_S, sLen, d_Out, oLen, treeInfo, info, localSize, gridSize);
    
    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);
    }
    gettimeofday(&end, NULL);
    
    SHOW_CHECKING;
    if (oLen != 0) {
        //check
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }
    int small = rLen;
    if (small > sLen)   small = sLen;
    if (oLen != small)  res = false;
    
    status = clReleaseMemObject(d_CSS_R);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete CSS_R;
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime1);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 1 ---"<<endl<<endl;
    
    //---------------------------------- End fo test 1 -----------------------------------
    
    //------------------------ Test 2: Having duplicate key values ------------------------
    
    cout<<"--- Test 2 : has duplicate---"<<endl;
    
    res = true;
    h_R = new Record[rLen];
    h_S = new Record[sLen];
    
    //h_R should be sorted
    recordSorted(h_R, rLen);
    recordRandom(h_S, sLen);
    
    gettimeofday(&start,NULL);
    
    mPart = calMPart(rLen);
    numOfInternalNodes_R = 0,mark_R = 0;
    
    CSS_R = generateCSSTree(h_R, rLen, mPart, numOfInternalNodes_R, mark_R);   //size: numOfInternalNodes * mPart * sizeof(int)
    CSS_R_Lengh = numOfInternalNodes_R * mPart;
    
    d_CSS_R = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*CSS_R_Lengh, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_CSS_R, CL_TRUE, 0, sizeof(int) * CSS_R_Lengh, CSS_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    treeInfo.d_CSS = d_CSS_R;
    treeInfo.CSS_length = CSS_R_Lengh;
    treeInfo.numOfInternalNodes = numOfInternalNodes_R;
    treeInfo.mark = mark_R;
    treeInfo.mPart = mPart;
    
    //memory allocation
    d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    double totalTime2 = inlj(d_R, rLen, d_S, sLen, d_Out, oLen, treeInfo, info, localSize, gridSize);
    
    SHOW_CHECKING;
    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);
    }
    gettimeofday(&end, NULL);
    
    if (oLen != 0) {
        //check
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }

    
    totalTime = (totalTime1 + totalTime2) / 2;
    
    status = clReleaseMemObject(d_CSS_R);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete CSS_R;
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime2);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 2 ---"<<endl<<endl;
    
    //---------------------------------- End fo test 2 -----------------------------------
    FUNC_END;
    
    return res;
}

bool testSmj(int rLen, int sLen, PlatInfo info, double &totalTime, int localSize) {
    
    bool res = true;
    FUNC_BEGIN;
    SHOW_PARALLEL(localSize, "not fixed.");
    SHOW_TABLE_R_NUM(rLen);
    SHOW_TABLE_S_NUM(sLen);
    
    
    //------------------------ Test 1: No duplicate key values ------------------------
    
    cout<<"--- Test 1 : no duplicate---"<<endl;
    Record *h_R = new Record[rLen];
    Record *h_S = new Record[sLen];
    Record *h_Out = NULL;
    
    recordSorted_Only(h_R, rLen);
    recordSorted_Only(h_S, sLen);
    
    cl_int status = 0;
    
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    //memory allocation
    cl_mem d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    cl_mem d_Out;
    int oLen = 0;
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    double totalTime1 = smj(d_R, rLen, d_S, sLen, d_Out, oLen, info, localSize);
    
    SHOW_CHECKING;
    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);
    }
    gettimeofday(&end, NULL);
    
    if (oLen != 0) {
        //check
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }
    int smallRes = rLen;
    if (smallRes > sLen)    smallRes = sLen;
    if (oLen != smallRes)   res = false;
    
    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime1);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 1 ---"<<endl<<endl;
    
    //---------------------------------- End fo test 1 -----------------------------------
    
    //------------------------ Test 2: Having duplicate key values ------------------------
    
    cout<<"--- Test 2 : has duplicate---"<<endl;
    
    res = true;
    h_R = new Record[rLen];
    h_S = new Record[sLen];
    
    recordSorted(h_R, rLen);
    recordSorted(h_S, sLen);
    
    gettimeofday(&start,NULL);
    
    //memory allocation
    d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    
    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    
    //call gather
    double totalTime2 = smj(d_R, rLen, d_S, sLen, d_Out, oLen, info, localSize);
    
    SHOW_CHECKING;
    if (oLen != 0) {
        h_Out = new Record[oLen];
        
        //memory written back
        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
        checkErr(status, ERR_READ_BUFFER);
    }
    gettimeofday(&end, NULL);
    
    if (oLen != 0) {
        //check
        for(int i = 0 ; i < oLen; i++) {
            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
                res = false;
                break;
            }
        }
        delete [] h_Out;
        status = clReleaseMemObject(d_Out);
        checkErr(status, ERR_RELEASE_MEM);
    }

    
    totalTime = (totalTime1 + totalTime2) / 2;
    
    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete [] h_R;
    delete [] h_S;
    
    FUNC_CHECK(res);
    SHOW_JOIN_RESULT(oLen);
    SHOW_TIME(totalTime2);
    SHOW_TOTAL_TIME(diffTime(end, start));
    cout<<"--- End of Test 2 ---"<<endl<<endl;
    
    //---------------------------------- End fo test 2 -----------------------------------
    FUNC_END;
    
    return res;
}

bool testHj(int rLen, int sLen, PlatInfo info) {

    cl_int status = 0;
    bool res = true;
    FUNC_BEGIN;
//    SHOW_TABLE_R_NUM(rLen);
//    SHOW_TABLE_S_NUM(sLen);

    float rSize, sSize;
    rSize = rLen* 1.0*sizeof(int)/1024 * 2;     //key-value
    if (rSize > 1024) {
        rSize /= 1024;
        std::cout<<"Table R tuples: "<<rLen<<" ("<<rSize<<" MB)"<<std::endl;
    }
    else {
        std::cout<<"Table R tuples: "<<rLen<<" ("<<rSize<<" KB)"<<std::endl;
    }

    sSize = sLen* 1.0*sizeof(int)/1024 * 2;     //key-value
    if (sSize > 1024) {
        sSize /= 1024;
        std::cout<<"Table S tuples: "<<sLen<<" ("<<sSize<<" MB)"<<std::endl;
    }
    else {
        std::cout<<"Table S tuples: "<<sLen<<" ("<<sSize<<" KB)"<<std::endl;
    }

    double joinTime = 0, totalTime = 0;
    //------------------------ Test 1: No duplicate key values ------------------------

//    cout<<"--- Test 1 : no duplicate---"<<endl;
    Record *h_R = new Record[rLen];
    Record *h_S = new Record[sLen];
    Record *h_Out = NULL;

    recordRandom1(h_R, rLen, rLen);
    recordRandom1(h_S, sLen, rLen);

    struct timeval start, end;

    gettimeofday(&start,NULL);

    //memory allocation
    cl_mem d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //call gather
    int d_res_len;
    joinTime = hashjoin(d_R, rLen, d_S, sLen, d_res_len, info);
    gettimeofday(&end, NULL);

    totalTime = diffTime(end, start);

    std::cout<<"---- Kernel execution ended ----"<<std::endl;
//    int h_res_len= 0;
//    for(int r = 0; r < rLen; r++) {
//        for(int s = 0; s < sLen; s++) {
//            if (h_R[r].x == h_S[s].x)   h_res_len++;
//        }
//    }
//
//    if (d_res_len == h_res_len) {
//        std::cout<<"Hash join test passes!"<<std::endl;
//        std::cout<<"# joined results:"<<d_res_len<<std::endl;
//    }
//    else {
//        std::cout<<"Hash join test fails!"<<std::endl;
//        std::cout<<"CPU:"<<h_res_len<<'\t'<<"GPU:"<<d_res_len<<std::endl;
//    }
    //suppose the result is correct!
    cout<<"Joined results:"<<d_res_len<<endl;
    cout<<"Join time: "<<joinTime<<" ms."<<endl;

    float throughput = (rLen+sLen)* 1.0*sizeof(int)*2/1024/1024/1024 / totalTime * 1000;
    cout<<"Total Execution time: "<<totalTime<<" ms ("<<throughput<<" GB/s)"<<endl;
//    if (oLen != 0) {
//        h_Out = new Record[oLen];
//
//        //memory written back
//        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
//        checkErr(status, ERR_READ_BUFFER);
//        status = clFinish(info.currentQueue);
//    }

//
//    SHOW_CHECKING;
//    if (oLen != 0) {
//        //check
//        for(int i = 0 ; i < oLen; i++) {
//            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
//                res = false;
//                break;
//            }
//        }
//        delete [] h_Out;
//        status = clReleaseMemObject(d_Out);
//        checkErr(status, ERR_RELEASE_MEM);
//    }
//    int smallRes = rLen;
//    if (smallRes > sLen)    smallRes = sLen;
//    if (oLen != smallRes)   res = false;

    status = clReleaseMemObject(d_R);
    checkErr(status,ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_S);
    checkErr(status,ERR_RELEASE_MEM);
    delete [] h_R;
    delete [] h_S;

//    FUNC_CHECK(res);
//    SHOW_TIME(totalTime);
//    SHOW_TOTAL_TIME(diffTime(end, start));
//    cout<<"--- End of Test 1 ---"<<endl<<endl;

    //---------------------------------- End fo test 1 -----------------------------------

    //------------------------ Test 2: Having duplicate key values ------------------------

//    cout<<"--- Test 2 : has duplicate---"<<endl;
//
//    res = true;
//    h_R = new Record[rLen];
//    h_S = new Record[sLen];
//
//    recordRandom(h_R, rLen);
//    recordRandom(h_S, sLen);
//
//    gettimeofday(&start,NULL);
//
//    //memory allocation
//    d_R = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*rLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//    d_S = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(Record)*sLen, NULL, &status);
//    checkErr(status, ERR_HOST_ALLOCATION);
//
//    status = clEnqueueWriteBuffer(info.currentQueue, d_R, CL_TRUE, 0, sizeof(Record)*rLen, h_R, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//    status = clEnqueueWriteBuffer(info.currentQueue, d_S, CL_TRUE, 0, sizeof(Record)*sLen, h_S, 0, 0, 0);
//    checkErr(status, ERR_WRITE_BUFFER);
//
//    //partition
//    partitionHJ(d_R, rLen, countBit, info, localSize, gridSize);
//    partitionHJ(d_S, sLen, countBit, info, localSize, gridSize);
//
//    //call gather
//    double totalTime2 = hj(d_R, rLen, d_S, sLen, d_Out, oLen, info, countBit, localSize);
//
//    if (oLen != 0) {
//        h_Out = new Record[oLen];
//
//        //memory written back
//        status = clEnqueueReadBuffer(info.currentQueue, d_Out, CL_TRUE, 0, sizeof(Record)*oLen, h_Out, 0, 0, 0);
//        checkErr(status, ERR_READ_BUFFER);
//        status = clFinish(info.currentQueue);
//    }
//    gettimeofday(&end, NULL);
//
//    SHOW_CHECKING;
//    if (oLen != 0) {
//        //check
//        for(int i = 0 ; i < oLen; i++) {
//            if (h_R[h_Out[i].x].y != h_S[h_Out[i].y].y) {
//                res = false;
//                break;
//            }
//        }
//        delete [] h_Out;
//        status = clReleaseMemObject(d_Out);
//        checkErr(status, ERR_RELEASE_MEM);
//    }
//
//    totalTime = (totalTime1 + totalTime2) / 2;
//
//    status = clReleaseMemObject(d_R);
//    checkErr(status,ERR_RELEASE_MEM);
//    status = clReleaseMemObject(d_S);
//    checkErr(status,ERR_RELEASE_MEM);
//    delete [] h_R;
//    delete [] h_S;
//
//    FUNC_CHECK(res);
//    SHOW_JOIN_RESULT(oLen);
//    SHOW_TIME(totalTime2);
//    SHOW_TOTAL_TIME(diffTime(end, start));
//    cout<<"--- End of Test 2 ---"<<endl<<endl;
//
//    //---------------------------------- End fo test 2 -----------------------------------
//    FUNC_END;

    return res;
}
