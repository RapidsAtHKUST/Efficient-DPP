/*
 * Execute on CPU:
 * 1. Set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. Compile the file using:
 *      icc -O3 -o scan_omp_cpu scan_omp.cpp -fopenmp
 * 3. Execute:
 *      ./scan_omp_cpu
 *
 * Execute on MIC (only native execution mode):
 * 1. Complile the file:
 *      icc -mmic -O3 -o scan_omp_mic scan_omp.cpp -fopenmp
 * 2. Copy the executable file to MIC:
 *      scp scan_omp_mic mic0:~
 * 3. (optional) If the MIC does not have libiomp5.so, copy the library from .../intel/lib/mic to MIC:
 *      e.g.: scp libiomp5.so mic0:~
 * 4. (optional) Set the library path on MIC:
 *      e.g.: export LD_LIBRARY_PATH=~
 * 5. Execute:
 *      ./scan_omp_mic
 */
#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>
#include "util/utility.h"
#include "util/log.h"
#include "util/timer.h"
#include "params.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/tick_count.h"

using namespace tbb;
#define MAX_THREAD_NUM (256)
using namespace std;

inline bool scan_check(int *input, int *output, uint64_t len) {
    int acc = 0;
    for (uint64_t i = 0; i < len; i++) {
        if (output[i] != acc) {
            log_error("Wrong result");
            return false;
        }
        acc += input[i];
    }
    return true;
}

/* TBB exclusive scan*/
template<typename T>
class ScanBody_ex {
    T sum;
    T* const y;
    const T* const x;
public:
    ScanBody_ex( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
    T get_sum() const {return sum;}

    template<typename Tag>
    void operator()( const blocked_range<int>& r, Tag ) {
        T temp = sum;
        int end = r.end();
        for( int i=r.begin(); i<end; ++i ) {
            if( Tag::is_final_scan() )
                y[i] = temp;
            temp = temp + x[i];
        }
        sum = temp;
    }
    ScanBody_ex( ScanBody_ex& b, split ) : x(b.x), y(b.y), sum(0) {}
    void reverse_join( ScanBody_ex& a ) { sum = a.sum + sum;}
    void assign( ScanBody_ex& b ) {sum = b.sum;}
};

double scan_tbb(int *input, int* output, uint64_t len) {
    Timer t;
    ScanBody_ex<int> body(output,input);
    parallel_scan(blocked_range<int>(0,len), body, auto_partitioner());
    return t.elapsed()*1000;
};

/*OpenMP-based scan-scan-add scheme, 4n data accesses*/
double scan_SSA_omp(int *input, int* output, uint64_t len) {
    int reduce_sum[MAX_THREAD_NUM] = {0};
    Timer t;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int local_sum = 0;

        /*Scan*/
#pragma omp for schedule(static) nowait
        for (int i = 0; i < len; i++) {
            output[i] = local_sum;
            local_sum += input[i];
        }
        reduce_sum[tid] = local_sum;
#pragma omp barrier

        /*Scan*/
#pragma omp single
        {
            int acc = 0;
            for (int i = 0; i < nthreads; i++) {
                int temp = reduce_sum[i];
                reduce_sum[i] = acc;
                acc += temp;
            }
        }

        /*Add*/
#pragma omp for schedule(static) nowait
        for (int i = 0; i < len; i++) {
            output[i] += reduce_sum[tid];
        }
    }
    return t.elapsed()*1000;
}

/*OpenMP-based reduce-then-scan scheme, 3n data accesses*/
double scan_RTS_omp(int *input, int* output, uint64_t len) {
    int reduce_sum[MAX_THREAD_NUM] = {0};
    Timer t;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int local_sum = 0;

        /*Reduce*/
#pragma omp for schedule(static) nowait
        for (int i = 0; i < len; i++) {
            local_sum += input[i];
        }
        reduce_sum[tid] = local_sum;
#pragma omp barrier

        /*Scan*/
#pragma omp single
        {
            int acc = 0;
            for (int i = 0; i < nthreads; i++) {
                int temp = reduce_sum[i];
                reduce_sum[i] = acc;
                acc += temp;
            }
        }

        /*Scan*/
        local_sum = reduce_sum[tid];
#pragma omp for schedule(static) nowait
        for (int i = 0; i < len; i++) {
            output[i] = local_sum;
            local_sum += input[i];
        }
    }
    return t.elapsed()*1000;
}

bool test_scan() {
    log_info("Function: %s", __FUNCTION__);
    int scale_min = 10, scale_max = 30;
    uint64_t max_len = pow(2, scale_max);
    double ave_time;
    bool res = true;

    int *input = new int[max_len];
    int *output = new int[max_len];
#pragma omp parallel for
    for(auto i = 0; i < max_len; i++) {
        input[i] = 1;
    }

    for(int scale = 10; scale <= 30; scale++) {
        int cur_len = 1<<scale;
        log_info("Current length = %d", cur_len);
        double tempTimes[EXPERIMENT_TIMES];

        /*SSA scan*/
        for(int e = 0; e < EXPERIMENT_TIMES; e++) {
            tempTimes[e] = scan_SSA_omp(input, output, cur_len);
            if (e == 0) res = scan_check(input, output, cur_len);
        }
        ave_time = average_Hampel(tempTimes, EXPERIMENT_TIMES);

        if (res) {
            log_info("SAA scan: time=%.1f ms, throughput=%.1f GB/s",
                     ave_time, compute_bandwidth(cur_len, sizeof(int), ave_time));
        }
        else break;

        /*RTS scan*/
        for(int e = 0; e < EXPERIMENT_TIMES; e++) {
            tempTimes[e] = scan_RTS_omp(input, output, cur_len);
            if (e == 0) res = scan_check(input, output, cur_len);
        }
        ave_time = average_Hampel(tempTimes, EXPERIMENT_TIMES);

        if (res) {
            log_info("RTS scan: time=%.1f ms, throughput=%.1f GB/s",
                     ave_time, compute_bandwidth(cur_len, sizeof(int), ave_time));
        }
        else break;

        /*TBB scan*/
        for(int e = 0; e < EXPERIMENT_TIMES; e++) {
            tempTimes[e] = scan_tbb(input, output, cur_len);
            if (e == 0) res = scan_check(input, output, cur_len);
        }
        ave_time = average_Hampel(tempTimes, EXPERIMENT_TIMES);

        if (res) {
            log_info("TBB scan: time=%.1f ms, throughput=%.1f GB/s",
                     ave_time, compute_bandwidth(cur_len, sizeof(int), ave_time));
        }
        else break;
    }
    return res;
}

int main(int argc, char* argv[]) {
    assert(test_scan());
    return 0;
}