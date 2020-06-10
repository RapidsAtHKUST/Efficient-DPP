//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "Plat.h"
#include "log.h"
using namespace std;

bool test_gather(int len) {
    log_trace("Function: %s", __FUNCTION__);
    bool res = true;
    cl_int status;
    auto param = Plat::get_device_param();

    /*kernel configurations*/
    auto local_size = 1024;
    auto elements_per_thread = 16;               //elements per thread
    auto grid_size = len / local_size / elements_per_thread;

    //data initialization
    int *h_in = new int[len];
    int *h_loc = new int[len];
    int *h_out = new int[len];
#pragma omp parallel for
    for(int i = 0; i < len; i++)    h_in[i] = i;
    log_info("Initializing data (%d items)...", len);
    random_generator_int_unique(h_loc, len);
    log_info("Initialization finished");

    cl_mem d_in = clCreateBuffer(param.context, CL_MEM_READ_ONLY, sizeof(int)*len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);
    cl_mem d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    cl_mem d_loc = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int)*len, nullptr, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(param.queue, d_loc, CL_TRUE, 0, sizeof(int)*len, h_loc, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    /*loop for multi-pass*/
    for(int pass = 1; pass <= 32 ; pass<<=1) {
        double myTime = 0;
        for(int i = 0; i < EXPERIMENT_TIMES; i++)  {
            double tempTime = gather(d_in, d_out, len, d_loc, local_size, grid_size, pass);
            myTime += tempTime;
        }
        myTime /= EXPERIMENT_TIMES;
        log_info("len=%d, size=%.1f MB, pass=%d, grid_size=%d, time=%.1f ms (%.1f ns/tuple, %.1f GB/s)",
                 len, 1.0*len* sizeof(int)/1024/1024, pass, grid_size, myTime, myTime/len*1e6, 1.0*len* sizeof(int)/myTime/1e6);
    }
    /*Load output back to host*/
    status = clEnqueueReadBuffer(param.queue, d_out, CL_TRUE, 0, sizeof(int) * len, h_out, 0, nullptr, nullptr);
    checkErr(status, ERR_READ_BUFFER);

    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_in);
    checkErr(status, ERR_RELEASE_MEM);
    status = clReleaseMemObject(d_loc);
    checkErr(status, ERR_RELEASE_MEM);

    /*check the results*/
    for(int i = 0; i < len; i++) {
        if (h_out[i] != h_in[h_loc[i]]) {
            res = false;
            log_warn("Incorrect results");
            break;
        }
    }
    if (res) log_info("Results check passed");

    delete[] h_loc;
    delete[] h_in;
    delete[] h_out;

    return res;
}

/*
 * Usage:
 *    ./test_gather DATA_CARDINALITY
 * */
int main(int argc, const char *argv[]) {
    Plat::plat_init();
    unsigned long long card = stoull(argv[1]);
    assert(test_gather(card));
    return 0;
}