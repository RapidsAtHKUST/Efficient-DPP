/*
 * Execute on CPU:
 * 1. set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. compile the file using:
 *      icc -O3 -o gather_scatter_CPU gather_scatter_CPU.cpp -fopenmp
 * 3. Execute:
 *      ./gather_scatter_CPU
 * To enable streaming store, modify the main function
 *
 */
#include <iostream>
#include <omp.h>
#include <cmath>
#include <immintrin.h>
#include <cassert>
#include "util/utility.h"
#include "util/log.h"
#include "util/timer.h"
#include "params.h"
using namespace std;

/*gather*/
double gather(int *input, int *output, int *idx, uint64_t len) {
    Timer t;
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++) {
        output[i] = input[idx[i]];
    }
    return t.elapsed()*1000;
}

/*scatter*/
double scatter(int *input, int *output, int *idx, uint64_t len) {
    Timer t;
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++) {
        output[idx[i]] = input[i];
    }
    return t.elapsed()*1000;
}

bool test_gather_and_scatter(uint64_t len) {
    log_info("Function: %s", __FUNCTION__);
    int *input = new int[len];
    int *idx = new int[len];
    int *output = new int[len];

    random_generator_int_unique(idx, len);
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++){
        input[i] = i;
    }

    double times[EXPERIMENT_TIMES];
    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        times[e] = gather(input, output, idx, len);

        if (e == 0) { /*check the outputs*/
            bool res = true;
            for(int i = 0; i < len; i++) {
                if(output[i] != input[idx[i]]) {
                    res = false;
                    break;
                }
            }
            if (!res)   log_error("Wrong results");
        }
    }
    double ave_time = average_Hampel(times, EXPERIMENT_TIMES);
    log_info("Performance of gather: time=%.1f ms, throughput=%.1f GB/s", ave_time, compute_bandwidth(len, sizeof(int), ave_time));

    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        times[e] = scatter(input, output, idx, len);

        if (e == 0) { /*check the outputs*/
            bool res = true;
            for(int i = 0; i < len; i++) {
                if(output[idx[i]] != input[i]) {
                    res = false;
                    break;
                }
            }
            if (!res) {
                log_error("Wrong results");
                return false;
            }
        }
    }
    ave_time = average_Hampel(times, EXPERIMENT_TIMES);
    log_info("Performance of scatter: time=%.1f ms, throughput=%.1f GB/s", ave_time, compute_bandwidth(len, sizeof(int), ave_time));

    if(input)  delete[] input;
    if(output)  delete[] output;
    if(idx)  delete[] idx;
    return true;
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    uint64_t len = stoull(argv[1]);
    assert(test_gather_and_scatter(len));
    return 0;
}