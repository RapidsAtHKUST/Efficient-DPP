//
//  TestPrimitives.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/21/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "Foundation.h"
#include <iomanip>
using namespace std;

void random_generator_int(int *in_keys, int length, int max) {
    sleep(1); srand((unsigned)time(NULL));
    for(int i = 0; i < length ; i++)    in_keys[i] = rand() % max;
}

void fixed_generator_int(int *in_keys, int length, int value) {
    for(int i = 0; i < length ; i++)    in_keys[i] = value;
}

void random_generator_tuples(tuple_t *in, int length, int max) {
    sleep(1); srand((unsigned)time(NULL));
    for(int i = 0; i < length ; i++)    {
        in[i].x = rand() % max;
        in[i].y = SPLIT_VALUE_DEFAULT;        /*all values set to 1*/
    }
}

void valRandom_Partitioned(int *arr, int length, int partitions) {
    srand((unsigned)time(NULL));
    sleep(1);

    //1. generate the partition array
    int *pars = new int[length];
    for(int i = 0; i < length; i++) pars[i] = rand()%partitions;

    //2.histogram
    int *his = new int[partitions];
    for(int i = 0; i < partitions; i++) his[i] = 0;
    for(int i = 0; i < length; i++) his[pars[i]]++;

    //3.scan exclusively
    int temp1 = 0;
    for(int i = 0; i < partitions; i++) {
        int temp2 = his[i];
        his[i] = temp1;
        temp1 += temp2;
    }

    //4. scatter
    for(int i = 0; i < length; i++) {
        arr[i] = his[pars[i]]++;
    }

    delete[] pars;
    delete[] his;
}

/*
 *  test cases:
 *  1. normal hashing using bit operations
 *  2. branching : # varies from 0 to 2
 *  3. trigonometric functions: float and double
 *  Together: 6 test cases
 */
bool testMap(PlatInfo& info, int repeat, int repeatTrans, int localSize, int gridSize) {
    
    bool res = true;
    cl_int status = 0;
    int experTime = 10;

    // timing variables
    double hash_time = 0 , float_time = 0, double_time = 0, float_blank_time = 0, double_blank_time = 0;

    int category = 35;
    double bTime[35];       //35 branches
    double bForTime[35];    //35 branches using for 

    // FUNC_BEGIN;
    // SHOW_PARALLEL(localSize, gridSize);
    // SHOW_DATA_NUM(length);
    
    //hashing & branching test together
    int length = localSize * gridSize * repeat;
//     SHOW_DATA_NUM(length);

    int *h_source_values = new int[length];
    int *h_dest_values = new int[length];    

    //initilize the input
    valRandom<int>(h_source_values, length, length);

    //memory allocation
    cl_mem d_source_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(int)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_dest_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_source_values, CL_TRUE, 0, sizeof(int)*length, h_source_values, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);  

//--------------------------------- branching test -------------------------
    // for(int branch = 1; branch <= category; branch++) {
    //     // cout<<branch<<endl;
    //     for(int i = 0; i < experTime; i++) {
    //         double tempTime = map_branching(d_source_values, d_dest_values, localSize, gridSize, info, repeat, branch);
    //         if (i != 0) {
    //             bTime[branch-1] += tempTime;
    //         }
    //     }

    //     //checking
    //     status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    //     checkErr(status, ERR_READ_BUFFER);

    //     for(int i = 0; i < length; i++) {
    //         int key = h_source_values[i] % branch;
    //         if (h_dest_values[i] != h_source_values[i] + key) {
    //             res = false;
    //             cerr<<"Wrong for branch "<<branch<<" in common branch."<<endl;
    //             cout<<h_dest_values[i]<<' '<<h_source_values[i]<<' '<<key<<' '<<h_source_values[i] + key<<endl;
    //             exit(1);
    //         }
    //     }
    //     bTime[branch-1] /= (experTime - 1);
    // }

    // for(int branch = 1; branch <= category; branch++) {
    //     // cout<<branch<<endl;
    //     for(int i = 0; i < experTime; i++) {
    //         double tempTime = map_branching_for(d_source_values, d_dest_values, localSize, gridSize, info, repeat, branch);
    //         if (i != 0) {
    //             bForTime[branch-1] += tempTime;
    //         }
    //     }

    //     //checking
    //     status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    //     checkErr(status, ERR_READ_BUFFER);

    //     for(int i = 0; i < length; i++) {
    //         int key = h_source_values[i] % branch;
    //         if (h_dest_values[i] != h_source_values[i] + key) {
    //             res = false;
    //             cerr<<"Wrong for branch "<<branch<<" in for branch."<<endl;
    //             cout<<h_dest_values[i]<<' '<<h_source_values[i]<<' '<<key<<' '<<h_source_values[i] + key<<endl;
    //             exit(1);
    //         }
    //     }
    //     bForTime[branch-1] /= (experTime - 1);
    // }

//--------------------------------- hashing test -------------------------
    //call map hashing
    for(int i = 0; i < experTime; i++) {
        double tempTime = map_hashing(d_source_values, d_dest_values, localSize, gridSize, info, repeat);
        if (i != 0) hash_time += tempTime;
    }
    hash_time /= (experTime - 1);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    //check for hash
    for(int i = 0; i < length; i++) {
        if ((h_source_values[i] & 0x3) != h_dest_values[i]) {
            res = false;
            break;
        }
    }
    FUNC_CHECK(res);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_dest_values, CL_TRUE, 0, sizeof(int)*length, h_dest_values, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    status = clReleaseMemObject(d_source_values);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_dest_values);
    checkErr(status, ERR_RELEASE_MEM);

    delete [] h_source_values;
    delete [] h_dest_values;

//--------------------- trigonometric test (float) -------------------------

    length = localSize * gridSize * repeatTrans;
    float *h_alpha = new float[length]; //varies from 0 to 2pi
    float *h_beta = new float[length];  //varies from -0.5pi to 0.5pi
    float *h_x = new float[length];
    float *h_y = new float[length];
    float *h_z = new float[length];

    const float r = 6371.004f;
    
    //initialize
    valRandom<float>(h_alpha, length, 2*PI);   // 0 -- 2pi
    valRandom<float>(h_beta, length, PI);      // 0 -- pi

    for(int i = 0; i < length; i++) {   // -0.5pi - 0.5pi
        h_beta[i] -= 0.5 * PI;
    }

    //memory allocation
    cl_mem d_alpha = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_beta = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(float)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_x = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(float)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_y = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(float)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    cl_mem d_z = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(float)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_alpha, CL_TRUE, 0, sizeof(float)*length, h_alpha, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    status = clEnqueueWriteBuffer(info.currentQueue, d_beta, CL_TRUE, 0, sizeof(float)*length, h_beta, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    //testing
    for(int i = 0; i < experTime; i++) {
        double tempTime, tempTime_blank;
        map_transform<float>(d_alpha, d_beta, r, d_x, d_y, d_z, localSize, gridSize, info, repeatTrans, tempTime_blank, tempTime);
        if (i != 0) {
            float_time += tempTime;
            float_blank_time += tempTime_blank;
        }
    }
    float_time /= (experTime - 1);
    float_blank_time /= (experTime - 1);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_x, CL_TRUE, 0, sizeof(int)*length, h_x, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clEnqueueReadBuffer(info.currentQueue, d_y, CL_TRUE, 0, sizeof(int)*length, h_y, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clEnqueueReadBuffer(info.currentQueue, d_z, CL_TRUE, 0, sizeof(int)*length, h_z, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    // cout<<"single:"<<endl;
    // for(int i = 0; i < 10; i++) {
    //     cout<<setprecision(20)<<h_alpha[i]<<' '<<h_beta[i]<<' '<<h_x[i]<<' '<<h_y[i]<<' '<<h_z[i]<<endl;
    // }

    status = clFinish(info.currentQueue);

    status = clReleaseMemObject(d_alpha);
    status = clReleaseMemObject(d_beta);
    status = clReleaseMemObject(d_x);
    status = clReleaseMemObject(d_y);
    status = clReleaseMemObject(d_z);
    checkErr(status, ERR_RELEASE_MEM);

    // delete[] h_alpha;
    // delete[] h_beta;
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;


//--------------------- trigonometric test (double) -------------------------

    length = localSize * gridSize * repeatTrans;
    double*h_alpha_d = new double[length]; //varies from 0 to 2pi
    double*h_beta_d = new double[length];  //varies from -0.5pi to 0.5pi
    double*h_x_d = new double[length];
    double*h_y_d = new double[length];
    double*h_z_d = new double[length];

    const double r_d = 6371.004;
    
    //initialize
    // valRandom<double>(h_alpha_d, length, 2*PI);   // 0 -- 2pi
    // valRandom<double>(h_beta_d, length, PI);      // 0 -- pi

    for(int i = 0; i < length; i++) {   // -0.5pi - 0.5pi
        // h_beta_d[i] -= 0.5 * PI;
        h_alpha_d[i] = (double)h_alpha[i];
        h_beta_d[i] = (double)h_beta[i];
    }

    //memory allocation
     d_alpha = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(double)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

     d_beta = clCreateBuffer(info.context, CL_MEM_READ_ONLY , sizeof(double)*length, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);

     d_x = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(double)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

     d_y = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(double)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

     d_z = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(double)*length,NULL , &status);
    checkErr(status, ERR_HOST_ALLOCATION);

    status = clEnqueueWriteBuffer(info.currentQueue, d_alpha, CL_TRUE, 0, sizeof(double)*length, h_alpha_d, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    status = clEnqueueWriteBuffer(info.currentQueue, d_beta, CL_TRUE, 0, sizeof(double)*length, h_beta_d, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER); 

    //testing
    for(int i = 0; i < experTime; i++) {
        double tempTime, tempTime_blank;
        map_transform<double>(d_alpha, d_beta, r, d_x, d_y, d_z, localSize, gridSize, info, repeatTrans, tempTime_blank, tempTime);
        if (i != 0) {
            double_time += tempTime;
            double_blank_time += tempTime_blank;
        }
    }
    double_time /= (experTime - 1);
    double_blank_time /= (experTime - 1);

    //memory written back
    status = clEnqueueReadBuffer(info.currentQueue, d_x, CL_TRUE, 0, sizeof(double)*length, h_x_d, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clEnqueueReadBuffer(info.currentQueue, d_y, CL_TRUE, 0, sizeof(double)*length, h_y_d, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clEnqueueReadBuffer(info.currentQueue, d_z, CL_TRUE, 0, sizeof(double)*length, h_z_d, 0, 0, 0);
    checkErr(status, ERR_READ_BUFFER);

    status = clFinish(info.currentQueue);

    // cout<<"double:"<<endl;
    // for(int i = 0; i < 10; i++) {
    //     cout<<setprecision(20)<<h_alpha_d[i]<<' '<<h_beta_d[i]<<' '<<h_x_d[i]<<' '<<h_y_d[i]<<' '<<h_z_d[i]<<endl;
    // }

    // cout<<"formal:";
    // cout<<r_d * sin(h_beta_d[9])<<endl;

    status = clReleaseMemObject(d_alpha);
    status = clReleaseMemObject(d_beta);
    status = clReleaseMemObject(d_x);
    status = clReleaseMemObject(d_y);
    status = clReleaseMemObject(d_z);
    checkErr(status, ERR_RELEASE_MEM);

    delete[] h_alpha;
    delete[] h_beta;
    delete[] h_alpha_d;
    delete[] h_beta_d;
    delete[] h_x_d;
    delete[] h_y_d;
    delete[] h_z_d;

    //summary
    cout<<setprecision(6)<<"Total hashing time: "<<hash_time<<" ms.\t Tuple num: "<<localSize * gridSize * repeat<<"\t Time per tuple: "<<hash_time / (localSize * gridSize * repeat) * 1e6<<" ns."<<endl;

    for(int i = 1; i <= category; i++) {
        cout<<"Total branch"<<i<<" time: "<<bTime[i-1]<<" ms.\t Tuple num: "<<localSize * gridSize * repeat<<"\t Time per tuple: "<<bTime[i-1] / (localSize * gridSize * repeat)* 1e6<<" ns."<<endl;
    }
 
    for(int i = 1; i <= category; i++) {
        cout<<"Total branch using for "<<i<<" time: "<<bForTime[i-1]<<" ms.\t Tuple num: "<<localSize * gridSize * repeat<<"\t Time per tuple: "<<bForTime[i-1] / (localSize * gridSize * repeat)* 1e6<<" ns."<<endl;
    }

    cout<<"Total trans_float time: "<<float_time<<" ms.\t Tuple num: "<<localSize * gridSize * repeatTrans<<"\t Time per tuple: "<<float_time / (localSize * gridSize * repeatTrans)* 1e6<<" ns."<<endl;

    cout<<"Total trans_float_blank time: "<<float_blank_time<<" ms.\t Tuple num: "<<localSize * gridSize * repeatTrans<<"\t Time per tuple: "<<float_blank_time / (localSize * gridSize * repeatTrans)* 1e6<<" ns."<<endl;

    cout<<"Total trans_double time: "<<double_time<<" ms.\t Tuple num: "<<localSize * gridSize * repeatTrans<<"\t Time per tuple: "<<double_time / (localSize * gridSize * repeatTrans)* 1e6<<" ns."<<endl;

     cout<<"Total trans_double_blank time: "<<double_blank_time<<" ms.\t Tuple num: "<<localSize * gridSize * repeatTrans<<"\t Time per tuple: "<<double_blank_time / (localSize * gridSize * repeatTrans)* 1e6<<" ns."<<endl;

     cout<<"For python:"<<endl;
     cout<<"mis = ["<<hash_time / (localSize * gridSize * repeat) * 1e6 <<','
         <<float_time / (localSize * gridSize * repeatTrans)* 1e6<<','
         <<float_blank_time / (localSize * gridSize * repeatTrans)* 1e6<<','
         <<double_time / (localSize * gridSize * repeatTrans)* 1e6<<','
         <<double_blank_time / (localSize * gridSize * repeatTrans)* 1e6<<']'<<endl;

    cout<<"category_common = ["<<bTime[0]/ (localSize * gridSize * repeat)* 1e6;
    for(int i = 1; i < category; i++) {
        cout<<','<<bTime[i]/ (localSize * gridSize * repeat)* 1e6;
    }
    cout<<"]"<<endl;

    cout<<"category_for = ["<<bForTime[0]/ (localSize * gridSize * repeat)* 1e6;
    for(int i = 1; i < category; i++) {
        cout<<','<<bForTime[i]/ (localSize * gridSize * repeat)* 1e6;
    }
    cout<<"]"<<endl;

    cout<<"For excel:"<<endl;
    cout<<"Common:"<<endl;
    for(int i = 0; i < category; i++) {
        cout<<bTime[i]/ (localSize * gridSize * repeat)* 1e6<<endl;
    }
    cout<<"For:"<<endl;
    for(int i = 0; i < category; i++) {
        cout<<bForTime[i]/ (localSize * gridSize * repeat)* 1e6<<endl;
    }

}

/*
 *  lengthMax:         maximum size
 */
bool testGather(int len, const PlatInfo info) {
    bool res = true;
    int exper_time = 10;
    cl_int status;

    //kernel configuration
    const size_t local_size = 1024;             //local size
    int elements_per_thread = 16;               //elements per thread
    int grid_size = len / local_size / elements_per_thread;

    //data initialization
    int *h_in = new int[len];     //no need to copy data
    int *h_loc = new int[len];
    for(int i = 0; i < len; i++)    h_in[i] = i;
    valRandom_Only(h_loc, len, len);
//    valRandom_Partitioned(h_loc, len, 8192);

    cl_mem d_in = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*len, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //loop for multi-pass
    for(int pass = 1; pass <= 32 ; pass<<=1) {
        cout<<"[Gather Running] len:"<<len<<' '
            <<"size:"<<len* sizeof(int)/1024/1024<<"MB "
            <<"pass:"<<pass<<' '
            <<"gridSize:"<<grid_size<<'\t';
        double myTime = 0;
        for(int i = 0; i < exper_time; i++)  {
            double tempTime = gather(d_in, d_out, len, d_loc, local_size, grid_size, info, pass);
            if (i != 0)   myTime += tempTime;
        }
        myTime /= (exper_time-1);
        cout<<"time:"<<myTime<<" ms.";
        cout<<"("<<myTime/len*1e6<<" ns/tuple, "<< len* sizeof(int)/myTime/1e6<<"GB/s)"<<endl;
    }

    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_in);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_loc;
    delete[] h_in;

    //check
    // for(int i = 0; i < length; i++) {
    //     if (h_dest_values[h_loc[i]] != h_source_values[i]) {
    //         res = false;
    //         break;
    //     }
    // }
    // FUNC_CHECK(res);

    return res;
}


/*
 *  lengthMax:         maximum size
 */
bool testScatter(int len, const PlatInfo info) {
    bool res = true;
    int exper_time = 10;
    cl_int status;

    //kernel configuration
    const size_t local_size = 1024;             //local size
    int elements_per_thread = 16;               //elements per thread
    int grid_size = len / local_size / elements_per_thread;

    //data initialization
    int *h_in = new int[len];     //no need to copy data
    int *h_loc = new int[len];
    for(int i = 0; i < len; i++)    h_in[i] = i;
    valRandom_Only(h_loc, len, len);
//    valRandom_Partitioned(h_loc, len, 8192);

    cl_mem d_in = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_loc, CL_TRUE, 0, sizeof(int)*len, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //loop for multi-pass
    for(int pass = 1; pass <= 32 ; pass<<=1) {
        cout<<"[Scatter Running] len:"<<len<<' '
            <<"size:"<<len* sizeof(int)/1024/1024<<"MB "
            <<"pass:"<<pass<<' '
            <<"gridSize:"<<grid_size<<'\t';
        double myTime = 0;
        for(int i = 0; i < exper_time; i++)  {
            double tempTime = scatter(d_in, d_out, len, d_loc, local_size, grid_size, info, pass);
            if (i != 0)   myTime += tempTime;
        }
        myTime /= (exper_time-1);
        cout<<"time:"<<myTime<<" ms.";
        cout<<"("<<myTime/len*1e6<<" ns/tuple, "<< len* sizeof(int)/myTime/1e6<<"GB/s)"<<endl;
    }

    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_in);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_loc;
    delete[] h_in;

    //check
    // for(int i = 0; i < length; i++) {
    //     if (h_dest_values[i] != h_source_values[h_loc[i]]) {
    //         res = false;
    //         break;
    //     }
    // }
    // FUNC_CHECK(res);

    return res;
}

//only test exclusive scan
bool testScan(int length, double &aveTime, int localSize, int gridSize, int R, int L, PlatInfo& info) {
    cl_int status = 0;
    bool res = true;

    float sizeMB = 1.0*length*sizeof(int)/1024/1024;
    cout<<"length:"<<length<<' ';

    int *h_input = new int[length];
    int *h_output = new int[length];
    srand(time(NULL));
//    for(int i = 0; i < length; i++) h_input[i] = rand() & 0xf;
    for(int i = 0; i < length; i++) h_input[i] = 1;

    int experTime = 20;
    double tempTimes[experTime];
    for(int e = 0; e < experTime; e++) {

        cl_mem d_inout = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int) * length, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(info.currentQueue, d_inout, CL_TRUE, 0, sizeof(int) * length, h_input, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
        clFinish(info.currentQueue);

        double tempTime = scan_fast(d_inout, length, info, localSize, gridSize, R, L);
        //three-kernel
//        double tempTime = scan_three_kernel(d_inout, length, info, localSize, gridSize);
//        double tempTime = scan_three_kernel_single(d_inout, length, info, gridSize);

        status = clEnqueueReadBuffer(info.currentQueue, d_inout, CL_TRUE, 0, sizeof(int) * length, h_output, 0, NULL, NULL);

        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(info.currentQueue);

        status = clReleaseMemObject(d_inout);
        checkErr(status, ERR_RELEASE_MEM);

        //check
        if (e == 0) {
            int acc = 0;
            for (int i = 0; i < length; i++) {
                if (h_output[i] != acc) {
                    res = false;
                    break;
                }
                acc += h_input[i];
//                cout<<h_output[i]<<' ';
            }
        }
        tempTimes[e] = tempTime;

    }
    aveTime = averageHampel(tempTimes, experTime);

    if(h_input) delete[] h_input;
    if(h_output) delete[] h_output;

//     cout<<"Time:"<<aveTime<<" ms.\t";
//     cout<<"Throughput:"<<sizeMB*1.0/1024/aveTime*1000<<" GB/s"<<endl;
    return res;
}

//search the most suitable (localSize, R, L) parameters for a scan scheme
/*
 * selection:
 *  0:GPU, 1:CPU, 2:MIC
 * restriction:
 * 1. GPU: local memory size: 48KB, register size: 64KB
 * 2. CPU: register size: 32KB
 */
void testScanParameters(int length, int device, PlatInfo& info) {
    int grid_size, size_begin, size_end, R_end, L_end, R_begin = 0, L_begin = 0;
    size_t cpu_reg = 32*1024, gpu_reg = 64*1024, gpu_local = 48*1024, mic_reg = 32*1024;

    if (device == 0) {       //gpu
        size_begin = 1024;
        size_end = 128;
        grid_size = 15;
    }
    else if (device == 1) {  //cpu, R and L share the registers
        size_begin = 512;
        size_end = 64;
        grid_size = 39;
    }
    else if (device == 2) { //MIC
        size_begin = 2048;
        size_end = 64;
        grid_size = 240;
    }

    int best_size, best_R, best_L;
    double bestThr = 0;
    for(size_t cur_size = size_begin; cur_size >= size_end; cur_size>>=1) {
        if (device == 0) {
            R_end = gpu_reg / cur_size / sizeof(int) - 1;
            L_end = gpu_local / cur_size / sizeof(int) - 1;
        }
        else if (device == 1) {
            R_end = cpu_reg / cur_size / sizeof(int) - 1;
            L_end = cpu_reg/ cur_size/ sizeof(int) - 1;
        }
        else if (device == 2) {
            R_end = mic_reg / cur_size / sizeof(int) - 1;
            L_end = 2;
        }
        for(int R = R_begin; R <= R_end; R++) {
            for(int L = L_begin; L <= L_end; L++) {
                if (R == 0 && L == 0) continue;
                if ( (device==1 || device==2) && (R+L>R_end))   continue;
                double tempTime;
                bool res = testScan(length,  tempTime, cur_size, grid_size, R, L, info);

                //compuation
                double throughput = 1.0*length/1024/1024/1024/tempTime*1000;
                cout<<"localSize="<<cur_size<<' '<<"R="<<R<<" L="<<L<<" Thr="<<throughput<<"GKeys/s";

                if (!res)   {
                    cout<<" wrong"<<endl;
                    break;
                }
                if (throughput>bestThr) {
                    bestThr = throughput;
                    best_size = cur_size;
                    best_R = R;
                    best_L = L;
                    cout<<" best!"<<endl;
                }
                else {
                    cout<<endl;
                }
            }
        }
    }
    cout<<"Final res:"<<endl;
    cout<<"\tBest localSize="<<best_size<<endl;
    cout<<"\tBest R="<<best_R<<endl;
    cout<<"\tBest L="<<best_L<<endl;
    cout<<"\tThroughput="<<bestThr<<"GB/s"<<endl;
}

//fanout: number of partitions needed
bool testSplit(int len, PlatInfo& info, int buckets, double& aveTime, int algo, Data_structure structure) {

    cl_int status = 0;
    bool res = true;
    int experTime = 1;
    double tempTime, *time_recorder = new double[experTime];

    cout<<"Length: "<<len<<'\t';
    cout<<"Buckets: "<<buckets<<"\t";

    int *h_in_keys=NULL, *h_in_values=NULL, *h_out_keys=NULL, *h_out_values=NULL;/*for SOA*/
    tuple_t *h_in=NULL, *h_out=NULL;      /*for AOS*/

    cl_mem d_in_keys=0, d_in_values=0, d_out_keys=0, d_out_values=0;
    cl_mem d_in=0, d_out=0;

    /*host memory allocation & initialization*/
    h_in_keys = new int[len];
    h_out_keys = new int[len];
    h_in_values = new int[len];
    h_out_values = new int[len];
    random_generator_int(h_in_keys, len, len);
    fixed_generator_int(h_in_values, len, SPLIT_VALUE_DEFAULT);  /*all values set to SPLIT_VALUE_DEFAULT*/

    if (structure == KVS_AOS) { /*KVS_AOS*/
        /*extra host memory initialization*/
        h_in = new tuple_t[len];
        h_out = new tuple_t[len];
        for(int i = 0; i < len; i++) {
            h_in[i].x = h_in_keys[i];
            h_in[i].y = h_in_values[i];
        }

        /*device memory initialization*/
        d_in = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(tuple_t)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(tuple_t)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        /*memory copy*/
        status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(tuple_t)*len, h_in, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
    }
    else {      /*KO or KVS_SOA*/
        /*device memory initialization*/
        d_in_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out_keys = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        /*memory copy*/
        status = clEnqueueWriteBuffer(info.currentQueue, d_in_keys, CL_TRUE, 0, sizeof(int)*len, h_in_keys, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);

        /*further initialize the values*/
        if(structure == KVS_SOA) {
            /*device memory initialization*/
            d_in_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);
            d_out_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            /*memory copy*/
            status = clEnqueueWriteBuffer(info.currentQueue, d_in_values, CL_TRUE, 0, sizeof(int)*len, h_in_values, 0, 0, 0);
            checkErr(status, ERR_WRITE_BUFFER);
        }
    }

    cl_mem d_in_unified=0, d_out_unified=0;
    if (structure == KVS_AOS)   {
        d_in_unified = d_in;
        d_out_unified = d_out;
    }
    else {
        d_in_unified = d_in_keys;
        d_out_unified = d_out_keys;
    }

    for(int e = 0; e < experTime; e++) {
        if (res == false)   break;
        switch (algo) {
            case 0:     /*key-value*/
                tempTime = single_split(d_in_unified, d_out_unified, len, buckets, false, structure, info);
                break;
            case 1:     /*key-value, reorder*/
                tempTime = single_split(d_in_unified, d_out_unified, len, buckets, true, structure, info);
                break;
            case 2:
                tempTime = WI_split(d_in_unified, d_out_unified, 0, len, buckets, structure, info, d_in_values, d_out_values);
                break;
            case 3:
                tempTime = WG_split(d_in_unified, d_out_unified, 0, len, buckets, true, structure, info, d_in_values, d_out_values);
                break;
        }

        if (e == 0) {
            if (structure == KVS_AOS) {
                status = clEnqueueReadBuffer(info.currentQueue, d_out, CL_TRUE, 0, sizeof(tuple_t)*len, h_out, 0, 0, 0);
                checkErr(status, ERR_READ_BUFFER);
                status = clFinish(info.currentQueue);
                /*initialize only for checking*/

                for(int i = 0; i < len; i++) {
                    h_out_keys[i] = h_out[i].x;
                    h_out_values[i] = h_out[i].y;
                }
            }
            else {
                status = clEnqueueReadBuffer(info.currentQueue, d_out_keys, CL_TRUE, 0, sizeof(int) * len, h_out_keys,
                                             0, 0, 0);
                checkErr(status, ERR_READ_BUFFER);
                if (structure == KVS_SOA) {
                    status = clEnqueueReadBuffer(info.currentQueue, d_out_values, CL_TRUE, 0, sizeof(int) * len,
                                                 h_out_values,
                                                 0, 0, 0);
                    checkErr(status, ERR_READ_BUFFER);
                }
            }
            status = clFinish(info.currentQueue);

            /*check the sum*/
            unsigned mask = buckets - 1;
            unsigned long check_total_in = 0;
            unsigned long check_total_out = 0;
            check_total_in += h_in_keys[0] & mask;
            check_total_out += h_out_keys[0] & mask;

            int bits_prev = h_out_keys[0] & mask;
            for(int i = 1; i < len; i++) {
                int bits_now = h_out_keys[i] & mask;
                check_total_out += bits_now;

                if (bits_now < bits_prev)  {
                    res = false;
                    std::cerr<<"wrong result, keys are not right!"<<std::endl;
                    break;
                }
                bits_prev = bits_now;

                /*accumulate the input data*/
                check_total_in += h_in_keys[i] & mask;
            }

            /*check the values*/
            if (structure != KO) {
                for(int i = 0; i < len; i++) {
                    if (h_out_values[i] != SPLIT_VALUE_DEFAULT) {
                        std::cout<<i<<' '<<h_in_values[i]<<' '<<h_out_values[i]<<' '<<SPLIT_VALUE_DEFAULT<<std::endl;
                        res = false;
                        std::cerr<<"wrong result, values are not right!"<<std::endl;
                        break;
                    }
                }
            }
            /*sum not equal*/
            if (check_total_in != check_total_out) {
                res = false;
                std::cerr<<"wrong result, key sum not match!"<<std::endl;
                std::cerr<<"right: "<<check_total_in<<"\toutput: "<<check_total_out<<std::endl;
                break;
            }
        }
        time_recorder[e] = tempTime;
    }
    aveTime = averageHampel(time_recorder, experTime);

    /*memory free*/
    cl_mem_free(d_in_keys);
    cl_mem_free(d_in_values);
    cl_mem_free(d_out_keys);
    cl_mem_free(d_out_values);
    cl_mem_free(d_in);
    cl_mem_free(d_out);

    if(h_in_keys)       delete [] h_in_keys;
    if(h_out_keys)      delete [] h_out_keys;
    if(h_in_values)     delete [] h_in_values;
    if(h_out_values)    delete [] h_out_values;
    if(h_in)            delete[] h_in;
    if(h_out)           delete[] h_out;
    if(time_recorder)   delete[] time_recorder;

    /*report the results*/
    std::cout<<"Time: "<<aveTime<<" ms."<<"("<<len* sizeof(int)*2/aveTime*1000/1e9<<"GB/s)"<<std::endl;

    return res;
}

//search the most suitable (localSize, gridSize, sharedMem size) parameters for a split scheme
/*
 * device:
 *  0:GPU, 1:CPU, 2:MIC
 * algo:
 *  0:thread_split_k
 *  1:thread_split_kv
 *  2:block_split_k (no reordering)
 *  3:block_split_k (reordering)
 *  4:block_split_kv (no reordering)
 *  5:block_split_kv (reordering)
 * restriction:
 * 1. GPU: local memory size: 48KB
 */
void testSplitParameters(int len, int buckets, int device, int algo, Data_structure structure, PlatInfo& info) {
    cl_int status;
    int experTime = 10;
    int localSizeBegin, localSizeEnd, gridSizeBegin, gridSizeEnd, limitSharedSize;

    //on gpu
    if (device==0) {
        localSizeBegin = 128;
        localSizeEnd = 1024;
        gridSizeBegin = 1024;
        gridSizeEnd = 131072;
        limitSharedSize = 47*1024;      //47KB
    }
    else if (device==1) {       //on CPU
        localSizeBegin = 64;
        localSizeEnd = 512;
        gridSizeBegin = 128;
        gridSizeEnd = 131072;
        limitSharedSize = 32*1024;    //CPU also has limited local mem size
    }
    else if (device==2) {       //on MIC
        localSizeBegin = 64;
        localSizeEnd = 2048;
        gridSizeBegin = 128;
        gridSizeEnd = 4096;
        limitSharedSize = 64*1024;    //CPU also has limited local mem size
    }
    else return;

    //best results
    double bestTime = 99999;
    int bestLocalSize=-1, bestGridSize=-1, bestSharedSize=-1;

    std::cout<<"Length="<<len<<" Algo="<<algo<<" Buckets="<<buckets<<' ';
    for(int localSize = localSizeBegin; localSize <= localSizeEnd; localSize<<=1) {
        for(int gridSize = gridSizeBegin; gridSize <= gridSizeEnd; gridSize <<=1) {
            int sharedSize = len / gridSize;

            //check the shared memory size
            if (sharedSize < localSize) continue;

            if (algo == 3 || algo == 5) {
                int scale = 0;
                if (algo == 3) scale = 1;
                if (algo == 5) scale = 2;

                if ( sizeof(int)*(buckets+1) + scale * sharedSize * sizeof(int) > limitSharedSize) {
                    continue;
                }
            }

            if (algo == 0 || algo == 1) {
                //check the local memory size for the thread-level split
                if (localSize * buckets * sizeof(int) >= limitSharedSize) continue;

                //check the global memory size for the thread-level split
                unsigned his_len = localSize * gridSize;
                unsigned limit = 268435456 / buckets;
                if (his_len > limit) continue;
            }

            bool res = true;

//------------------ data initialization -------------------
            int *h_in_keys = new int[len];
            int *h_out_keys = new int[len];

            int *h_in_values = new int[len];
            int *h_out_values = new int[len];

            recordRandom(h_in_keys, h_in_values, len, len);
            for(int i = 0; i < len; i++) h_in_values[i] = h_in_keys[i];

            //memory allocation
            cl_mem d_in_keys = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);
            //output data
            cl_mem d_out_keys = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            cl_mem d_in_values = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);
            cl_mem d_out_values = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            //data copy
            status = clEnqueueWriteBuffer(info.currentQueue, d_in_keys, CL_TRUE, 0, sizeof(int)*len, h_in_keys, 0, 0, 0);
            checkErr(status, ERR_WRITE_BUFFER);
            status = clEnqueueWriteBuffer(info.currentQueue, d_in_values, CL_TRUE, 0, sizeof(int)*len, h_in_values, 0, 0, 0);
            checkErr(status, ERR_WRITE_BUFFER);
//------------------ data initialization end-------------------
            float aveTime = 0.0;
            double *timeRecorder = new double[experTime];
            for(int e = 0; e < experTime; e++) {
                double tempTime;
                switch (algo) {
                    case 0:
                        tempTime = WI_split(d_in_keys, d_out_keys, NULL, len, buckets, structure, info, NULL, NULL, localSize, gridSize);
                        break;
                    case 1:
                        tempTime = WI_split(d_in_keys, d_out_keys, NULL, len, buckets, structure, info, d_in_values, d_out_values, localSize, gridSize);
                        break;
                    case 2:
                        tempTime = WG_split(d_in_keys, d_out_keys, NULL, len, buckets, false, structure, info, NULL, NULL, localSize, gridSize);
                        break;
                    case 3:
                        tempTime = WG_split(d_in_keys, d_out_keys, NULL, len, buckets, true, structure, info, NULL, NULL,localSize, gridSize);
                        break;
                    case 4:
                        tempTime = WG_split(d_in_keys, d_out_keys, NULL, len, buckets, false, structure, info, d_in_values, d_out_values, localSize, gridSize);
                        break;
                    case 5:
                        tempTime = WG_split(d_in_keys, d_out_keys, NULL, len, buckets, true, structure, info, d_in_values, d_out_values, localSize, gridSize);
                        break;
                }
                if (e == 0) {
                    status = clEnqueueReadBuffer(info.currentQueue, d_out_keys, CL_TRUE, 0, sizeof(int)*len, h_out_keys, 0, 0, 0);
                    checkErr(status, ERR_READ_BUFFER);
                    status = clFinish(info.currentQueue);

                    int mask = buckets - 1;
                    int bits_prev = h_out_keys[0] & mask;
                    for(int i = 1; i < len; i++) {
                        int bits_now = h_out_keys[i] & mask;
                        if (bits_now < bits_prev)  {
                            res = false;
                            cerr<<"\twrong results"<<endl;
                            return ;
                        }
                        bits_prev = bits_now;
                    }
                }
                timeRecorder[e] = tempTime;
            }
            aveTime = averageHampel(timeRecorder,experTime);
            if (timeRecorder)   delete[] timeRecorder;

            if (aveTime < bestTime) {
                bestTime = aveTime;
                bestLocalSize = localSize;
                bestGridSize = gridSize;
                bestSharedSize = sharedSize;
            }

            status = clReleaseMemObject(d_in_keys);
            checkErr(status, ERR_RELEASE_MEM);
            status = clReleaseMemObject(d_out_keys);
            checkErr(status, ERR_RELEASE_MEM);
            status = clReleaseMemObject(d_in_values);
            checkErr(status, ERR_RELEASE_MEM);
            status = clReleaseMemObject(d_out_values);
            checkErr(status, ERR_RELEASE_MEM);

            if(h_in_keys) delete [] h_in_keys;
            if(h_out_keys) delete [] h_out_keys;
            if(h_in_values) delete [] h_in_values;
            if(h_out_values) delete [] h_out_values;
        }
    }
    cout<<"bLocal="<<bestLocalSize<<" bGrid="<<bestGridSize<<" Time="<<bestTime<<"ms"<<endl;
}

bool testRadixSort(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, 
    int length, PlatInfo& info, double& totalTime) {
    
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

bool testBitonitSort(Record *fixedSource, int length, PlatInfo& info, int dir, double& totalTime, int localSize, int gridSize) {
    
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


