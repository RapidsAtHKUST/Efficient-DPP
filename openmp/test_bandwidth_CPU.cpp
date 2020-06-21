/*
 * Execute on CPU:
 * 1. set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. compile the file using:
 *      icc -O3 -o mem_cpy_cpu mem_cpy.cpp -fopenmp
 * 3. Execute:
 *      ./cpy_omp_cpu
 * To enable streaming store, modify the main function
 *
 * Execute on MIC (only native execution mode):
 * 1. Compile the file:
 *      icc -mmic -O3 -o mem_cpy_mic mem_cpy.cpp -fopenmp
 * 1.5 Compile with Streaming Store:
 *      icc -mmic -O3 -o mem_cpy_mic_ss mem_cpy.cpp -fopenmp -qopt-streaming-stores always
 * 2. Copy the executable file to MIC:
 *      scp mem_cpy_mic mic0:~
 * 3. (optional) If the MIC does not have libiomp5.so, copy the library from .../intel/lib/mic to MIC:
 *      e.g.: scp libiomp5.so mic0:~
 * 4. (optional) Set the library path on MIC:
 *      e.g.: export LD_LIBRARY_PATH=~
 * 5. Execute:
 *      ./mem_cpy_mic
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

#define SCALAR  (3)

/* Sequential memory copy operation */
double copy_omp(int *input, int *output, uint64_t len) {
    Timer t;
#pragma omp parallel for schedule(static)
    for(uint64_t i = 0; i < len; i++) {
        output[i] = input[i];
    }
    return t.elapsed()*1000; //in ms
}

/* Sequential scaling operation */
double scale_omp(int *input, int *output, uint64_t len) {
    Timer t;
#pragma omp parallel for schedule(static)
    for(uint64_t i = 0; i < len; i++) {
        output[i] = input[i] * SCALAR;
    }
    return t.elapsed()*1000; //in ms
}

/* Sequential memory copy operation with nontemporal streaming stores */
double copy_omp_ss(int *input, int *output, uint64_t len) {
    Timer t;
#pragma omp parallel for schedule(auto)
    for(uint64_t i = 0; i < len/8; i++) { //256-bit = 8 int values
        register __m256i *dest = (__m256i*)output + i;
        register __m256i source = *((__m256i*)input + i);
        _mm256_stream_si256(dest,source);   //streaming store
    }
    return t.elapsed()*1000; //in ms
}

/* Sequential memory copy operation with nontemporal streaming stores */
double scale_omp_ss(int *input, int *output, uint64_t len) {
    __m256i v = _mm256_set_epi32(SCALAR,SCALAR,SCALAR,SCALAR,
                                 SCALAR,SCALAR,SCALAR,SCALAR);
    Timer t;
#pragma omp parallel for schedule(auto)
    for(uint64_t i = 0; i < len/8; i++) { //256-bit = 8 int values
        register __m256i *dest = (__m256i*)output + i;
        register __m256i source = *((__m256i*)input + i);
        source = _mm256_mullo_epi32 (source, v);
        _mm256_stream_si256(dest,source);   //streaming store
    }
    return t.elapsed()*1000; //in ms
}

bool test_bandwidth(int max_len_log) {
    log_info("Function: %s", __FUNCTION__);
    assert(max_len_log > 10);
    uint64_t max_len = pow(2, max_len_log);

    int *input = new int[max_len];
    int *output = new int[max_len];
    int *input_aligned = (int*)_mm_malloc(sizeof(int)*max_len, 64);
    int *output_aligned = (int*)_mm_malloc(sizeof(int)*max_len, 64);

#pragma omp parallel for
    for(int i = 0; i < max_len; i++) input[i] = i;

    /* copy operation without streaming stores*/
    log_info("Copy operation without Streaming Stores");
    for(auto len_log = 10; len_log < max_len_log; len_log++) {
        uint64_t len = pow(2,len_log);
        double times[EXPERIMENT_TIMES];
        for(auto e = 0; e < EXPERIMENT_TIMES; e++) {
            times[e] = copy_omp(input, output, len);
        }
        auto ave_time = average_Hampel(times, EXPERIMENT_TIMES);
        log_info("Len=%d, time=%.1f ms, throughput=%.1f GB/s",
        len, ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));
    }

    /* scale operation without streaming stores*/
    log_info("Scale operation without Streaming Stores");
    for(auto len_log = 10; len_log < 30; len_log++) {
        uint64_t len = pow(2,len_log);
        double times[EXPERIMENT_TIMES];
        for(auto e = 0; e < EXPERIMENT_TIMES; e++) {
            times[e] = scale_omp(input, output, len);
        }
        auto ave_time = average_Hampel(times, EXPERIMENT_TIMES);
        log_info("Len=%d, time=%.1f ms, throughput=%.1f GB/s",
                 len, ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));
    }

    /* copy operation without streaming stores*/
    log_info("Copy operation with Streaming Stores");
    for(auto len_log = 10; len_log < 30; len_log++) {
        uint64_t len = pow(2,len_log);
        double times[EXPERIMENT_TIMES];
        for(auto e = 0; e < EXPERIMENT_TIMES; e++) {
            times[e] = copy_omp_ss(input_aligned, output_aligned, len);
        }
        auto ave_time = average_Hampel(times, EXPERIMENT_TIMES);
        log_info("Len=%d, time=%.1f ms, throughput=%.1f GB/s",
                 len, ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));
    }

    /* scale operation without streaming stores*/
    log_info("Scale operation with Streaming Stores");
    for(auto len_log = 10; len_log < 30; len_log++) {
        uint64_t len = pow(2,len_log);
        double times[EXPERIMENT_TIMES];
        for(auto e = 0; e < EXPERIMENT_TIMES; e++) {
            times[e] = scale_omp_ss(input_aligned, output_aligned, len);
        }
        auto ave_time = average_Hampel(times, EXPERIMENT_TIMES);
        log_info("Len=%d, time=%.1f ms, throughput=%.1f GB/s",
                 len, ave_time, compute_bandwidth(len*2, sizeof(int), ave_time));
    }

    delete[] input;
    delete[] output;
    _mm_free(input_aligned);
    _mm_free(output_aligned);

    return true;
}

int main(int argc, char* argv[]) {
    assert(test_bandwidth(30));
    return 0;
}