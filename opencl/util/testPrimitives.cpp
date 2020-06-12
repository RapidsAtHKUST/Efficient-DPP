//
//  TestPrimitives.cpp
//  gpuqp_opencl
//
//  Created by Zhuohang Lai on 5/21/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#include "Plat.h"
#include "utility.h"
using namespace std;

void fixed_generator_int(int *in_keys, uint64_t length, int value) {
    for(int i = 0; i < length ; i++)    in_keys[i] = value;
}

void random_generator_tuples(tuple_t *in, uint64_t length, int max) {
    sleep(1); srand((unsigned)time(NULL));
    for(ulong i = 0; i < length ; i++)    {
        in[i].x = rand() % max;
        in[i].y = SPLIT_VALUE_DEFAULT;        /*all values set to 1*/
    }
}

void valRandom_Partitioned(int *arr, uint64_t length, int partitions) {
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
 * Split test inner function, to test specific kernel configurations
 *
 *  len:            length of the dataset
 *  algo:           WI_split, WG_split, WG_reorder_split, Single_split, Single_reorder_split
 *  structure:      KO, AOS or SOA
 *
 * */
bool split_test(
        int len, int buckets, double& ave_time,
        Algo algo, Data_structure structure,
        int local_size, int grid_size)
{
    device_param_t param = Plat::get_device_param();

    cl_int status = 0;
    bool res = true;
    int experTime = 10;
    double tempTime, *time_recorder = new double[experTime];

    int *h_in_keys=NULL, *h_in_values=NULL, *h_out_keys=NULL, *h_out_values=NULL;/*for SOA*/
    tuple_t *h_in=NULL, *h_out=NULL;      /*for AOS*/

    cl_mem d_in_keys=0, d_in_values=0, d_out_keys=0, d_out_values=0;
    cl_mem d_in=0, d_out=0;

    /*host memory allocation & initialization*/
    h_in_keys = new int[len];
    h_out_keys = new int[len];
    h_in_values = new int[len];
    h_out_values = new int[len];
    random_generator_int(h_in_keys, len, len, 1234);
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
        d_in = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(tuple_t)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(tuple_t)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        /*memory copy*/
        status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(tuple_t)*len, h_in, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
    }
    else {      /*KO or KVS_SOA*/
        /*device memory initialization*/
        d_in_keys = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out_keys = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        /*memory copy*/
        status = clEnqueueWriteBuffer(param.queue, d_in_keys, CL_TRUE, 0, sizeof(int)*len, h_in_keys, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);

        /*further initialize the values*/
        if(structure == KVS_SOA) {
            /*device memory initialization*/
            d_in_values = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);
            d_out_values = clCreateBuffer(param.context, CL_MEM_WRITE_ONLY, sizeof(int)*len, NULL, &status);
            checkErr(status, ERR_HOST_ALLOCATION);

            /*memory copy*/
            status = clEnqueueWriteBuffer(param.queue, d_in_values, CL_TRUE, 0, sizeof(int)*len, h_in_values, 0, 0, 0);
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
            case WI:     /*WI-level split*/
                tempTime = WI_split(
                        d_in_unified, d_out_unified, 0,
                        len, buckets, structure,
                        d_in_values, d_out_values,
                        local_size, grid_size);
                break;
            case WG:     /*WG-level split*/
                tempTime = WG_split(
                        d_in_unified, d_out_unified, 0,
                        len, buckets, NO_REORDER, structure,
                        d_in_values, d_out_values,
                        local_size, grid_size);
                break;
            case WG_varied_reorder:     /*WG-level split, reorder*/
                tempTime = WG_split(
                        d_in_unified, d_out_unified, 0,
                        len, buckets, VARIED_REORDER, structure,
                        d_in_values, d_out_values,
                        local_size, grid_size);
                break;
            case WG_fixed_reorder:     /*WG-level split, reorder*/
                tempTime = WG_split(
                        d_in_unified, d_out_unified, 0,
                        len, buckets, FIXED_REORDER, structure,
                        d_in_values, d_out_values,
                        local_size, grid_size);
                break;
            case Single:     /*WG-level split, reorder*/
                tempTime = single_split(
                        d_in_unified, d_out_unified,
                        len, buckets, false, structure);
                break;
            case Single_reorder:     /*WG-level split, reorder*/
                tempTime = single_split(
                        d_in_unified, d_out_unified,
                        len, buckets, true, structure);
                break;
        }

        /*check the result*/
        if (e == 0) {
            if (structure == KVS_AOS) {
                status = clEnqueueReadBuffer(
                        param.queue, d_out, CL_TRUE, 0,
                        sizeof(tuple_t)*len, h_out, 0, 0, 0);
                checkErr(status, ERR_READ_BUFFER);
                status = clFinish(param.queue);
                /*initialize only for checking*/

                for(int i = 0; i < len; i++) {
                    h_out_keys[i] = h_out[i].x;
                    h_out_values[i] = h_out[i].y;
                }
            }
            else {
                status = clEnqueueReadBuffer(
                        param.queue, d_out_keys, CL_TRUE, 0,
                        sizeof(int) * len, h_out_keys, 0, 0, 0);
                checkErr(status, ERR_READ_BUFFER);
                if (structure == KVS_SOA) {
                    status = clEnqueueReadBuffer(
                            param.queue, d_out_values, CL_TRUE, 0,
                            sizeof(int) * len, h_out_values, 0, 0, 0);
                    checkErr(status, ERR_READ_BUFFER);
                }
            }
            status = clFinish(param.queue);

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
    ave_time = average_Hampel(time_recorder, experTime);

    /*memory free*/
    cl_mem_free(d_in_keys);
    cl_mem_free(d_in_values);
    cl_mem_free(d_out_keys);
    cl_mem_free(d_out_values);
    cl_mem_free(d_in);
    cl_mem_free(d_out);

    if(h_in_keys)       delete[] h_in_keys;
    if(h_out_keys)      delete[] h_out_keys;
    if(h_in_values)     delete[] h_in_values;
    if(h_out_values)    delete[] h_out_values;
    if(h_in)            delete[] h_in;
    if(h_out)           delete[] h_out;
    if(time_recorder)   delete[] time_recorder;

    return res;
}

void split_test_specific(
        int len, int buckets,
        Algo algo, Data_structure structure,
        int local_size, int grid_size)
{
    cout<<"Length: "<<len<<'\t'
        <<"Buckets: "<<buckets<<"\t";

    double my_time;
    bool res = split_test(len, buckets, my_time, algo, structure, local_size, grid_size);

    /*report the results*/
    if (res) {
        double throughput = compute_bandwidth(len, sizeof(int), my_time);
        cout << "Time: " << my_time << " ms." << "(" << throughput << "GB/s)" << endl;
    }
    else {
        cerr<<"Wrong answer."<<endl;
    }
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
void split_test_parameters(
        int len, int buckets,
        Algo algo, Data_structure structure,
        int device)
{
    device_param_t param = Plat::get_device_param();

    cl_int status;
    int local_size_begin, local_size_end, grid_size_begin, grid_size_end, local_mem_limited;

    /*Single split has fixed parameters*/
    if (algo == WG_fixed_reorder) {
        std::cout<<"Single split has fixed parameter: "
                 <<"local_size=1, grid_size=#CUs. "
                 <<"No need to probe."<<std::endl;
        return;
    }

    //on gpu
    if (device==0) {
        local_size_begin = 128;
        local_size_end = 1024;
        grid_size_begin = 1024;
        grid_size_end = 131072;
        local_mem_limited = 47*1024;      //47KB
    }
    else if (device==1) {       //on CPU
        local_size_begin = 1;
        local_size_end = 1;
        grid_size_begin = 4096;
        grid_size_end = 4096;
        local_mem_limited = 32*1024;    //CPU also has limited local mem size
    }
    else if (device==2) {       //on MIC
        local_size_begin = 1;
        local_size_end = 1;
        grid_size_begin = 256;
        grid_size_end = 32768;
        local_mem_limited = 32*1024;    //MIC also has limited local mem size
    }
    else return;

    /*best result*/
    double best_time = 99999;
    int local_size_best=-1, grid_size_best=-1, local_mem_size_best=-1;

    std::cout<<"Length="<<len<<" Algo="<<algo<<" Buckets="<<buckets<<' ';
    for(int local_size = local_size_begin; local_size <= local_size_end; local_size<<=1) {
        for(int grid_size = grid_size_begin; grid_size <= grid_size_end; grid_size <<=1) {
            int local_buffer_len = len / grid_size;

            //check the shared memory size
            if (local_buffer_len < local_size) continue;

            if (algo == WG_varied_reorder) {
                int scale = 0;
                if (structure == KO)    scale = 1;
                else                    scale = 2;

                /*local memory size exceeds the limit*/
                size_t local_size_used = sizeof(int)*(1+buckets+scale*local_buffer_len);
                if (local_size_used > local_mem_limited)    continue;
            }

            if (algo == WI) {
                /*check the local memory size for the thread-level split*/
                if (local_size * buckets * sizeof(int) >= local_mem_limited) continue;

                /*check the global memory size for the thread-level split*/
                unsigned his_len = local_size * grid_size;
                unsigned limit = 268435456 / buckets;
                if (his_len > limit) continue;
            }

            /*invode split_test
             *with the configuration*/
            double temp_time;
            bool res = split_test(
                    len, buckets, temp_time,
                    algo, structure,
                    local_size, grid_size);

//            cout<<"Time:"<<temp_time<<" lsize:"<<local_size<<" gsize:"<<grid_size<<endl;

            if (temp_time < best_time) {
                best_time = temp_time;
                local_size_best = local_size;
                grid_size_best = grid_size;
                local_mem_size_best = local_buffer_len;

            }
        }
    }
    cout<<"bLocal="<<local_size_best<<" bGrid="<<grid_size_best<<" Time="<<best_time<<"ms"<<endl;
}


