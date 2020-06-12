//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "Plat.h"
#include "log.h"
#include "../types.h"

bool test_scan(int len, double &ave_time, int local_size, int grid_size, scan_arg arg) {
    log_trace("Function: %s", __FUNCTION__);
    cl_int status = 0;
    bool res = true;
    device_param_t param = Plat::get_device_param();

    int *h_input = new int[len];
    int *h_output = new int[len];
    cl_mem d_in, d_out;

    srand(time(nullptr));
    log_info("Initializing data (%d items)...", len);
    for(int i = 0; i < len; i++) h_input[i] = rand() & 0xf;
    log_info("Initialization finished");

    auto R = arg.R;
    auto L = arg.L;
    auto algo = arg.algo;
    double record_times[EXPERIMENT_TIMES], cur_time;
    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        d_in = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * len, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * len, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int) * len, h_input, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
        clFinish(param.queue);

        switch (algo) {
            case RSS: {
                cur_time = scan_RSS(d_in, d_out, len, local_size, grid_size);
                break;
            }
            case RSS_SINGLE_THREAD: {
                cur_time = scan_RSS_single(d_in, d_out, len);
                break;
            }
            case CHAINED: {
                cur_time = scan_chained(d_in, d_out, len, local_size, grid_size, R, L);
                break;
            }
        }
        status = clEnqueueReadBuffer(param.queue, d_out, CL_TRUE, 0, sizeof(int) * len,
                                     h_output, 0, nullptr, nullptr);

        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(param.queue);
        status = clReleaseMemObject(d_in);
        checkErr(status, ERR_RELEASE_MEM);
        status = clReleaseMemObject(d_out);
        checkErr(status, ERR_RELEASE_MEM);

        /*check*/
        if (e == 0) {
            int acc = 0;
            for (int i = 0; i < len; i++) {
                if (h_output[i] != acc) {
                    log_warn("Wrong result: %d: %d in out, should be %d", i, h_output[i], acc);
                    res = false;
                    break;
                }
                acc += h_input[i];
            }
        }
        if(!res) break;
        record_times[e] = cur_time;
    }
    if(h_input) delete[] h_input;
    if(h_output) delete[] h_output;

    if (!res) return res;
    ave_time = average_Hampel(record_times, EXPERIMENT_TIMES);
    return res;
}

int main(int argc, char* argv[]) {
    Plat::plat_init();
    scan_arg args{0, 11, CHAINED}; //Best setting for GPU
//    scan_arg args{112, 0, CHAINED}; //Best setting for CPU
//    scan_arg args{67, 0, CHAINED}; //Best setting for MIC

    /*kernel setting*/
    int local_size = 1024, grid_size = 80; //for GPU
//    int local_size = 64, grid_size = 40; //for CPU
//    int local_size = 64, grid_size = 240; //for MIC

    for(int scale = 10; scale <= 30; scale++) {
        int len = 1<<scale;
        double ave_time;
        bool res = test_scan(len, ave_time, local_size, grid_size, args);
        if (!res) {
            log_error("Wrong result");
            exit(1);
        }
        log_info("Scale=%d, time=%.1f ms, throughput=%.1f GKeys/s",
                 scale, ave_time, compute_bandwidth(len, 1, ave_time));
    }
    return 0;
}
