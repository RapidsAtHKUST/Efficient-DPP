#pragma offload_attribute(push, target(mic))
#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_sort.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_scan.h"
#pragma offload_attribute(pop)

#include "functions.h"

using namespace std;
using namespace tbb;

#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE-1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)

#define MAX_THREAD_NUM      (256)

template<typename T> class __attribute__ ((target(mic))) ScanBody_ex {
    T sum;
    T* const y;
    const T* const x; 
public:
    ScanBody_ex( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
    T get_sum() const {return sum;}
 
    template<typename Tag>
    void operator()( const blocked_range<int>& r, Tag ) {
        T temp = sum;
        for( int i=r.begin(); i<r.end(); ++i ) {
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

double radixSort(unsigned *data, size_t n) {

    __attribute__ ((target(mic)))  unsigned *buffer;
    int total_digits = sizeof(unsigned)*8;
 
    // kmp_set_defaults("KMP_AFFINITY=compact");
    // kmp_set_defaults("KMP_BLOCKTIME=0");

    struct timeval start, end;

    //device memory allocation
    #pragma offload target(mic) \
    nocopy(buffer:length(n) alloc_if(1) free_if(0)) \
    in(data:length(n) alloc_if(1) free_if(0))
    {};

    gettimeofday(&start, NULL);
    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {

        #pragma offload target(mic) nocopy(data) nocopy(buffer:length(n))
        {
            size_t bucket[BASE * MAX_THREAD_NUM] = {0};
            size_t bucket_scanned[BASE * MAX_THREAD_NUM] = {0};

            //done by all the threads
            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();
                
                size_t local_bucket[BASE] = {0};    //local array

                #pragma omp for schedule(static) nowait
                for(int i = 0; i < n; i++){
                    local_bucket[DIGITS(data[i], shift)]++;
                }
                for(int i = 0; i < BASE; i++) {
                    bucket[nthreads * i + tid] = local_bucket[i];
                }
            }

            ScanBody_ex<size_t> body(bucket_scanned,bucket);
            parallel_scan( blocked_range<int>(0,MAX_THREAD_NUM * BASE), body );

            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                size_t local_bucket[BASE] = {0};    //local array

                for(int i = 0; i < BASE; i++) {
                     local_bucket[i] = bucket_scanned[nthreads * i + tid];
                }
                #pragma omp barrier

                #pragma omp for schedule(static)
                for(int i = 0; i < n; i++) { //note here the end condition
                    buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];
                }
            }

            //done by thread 0
            unsigned* tmp = data;
            data = buffer;
            buffer = tmp;
        }
    }
    gettimeofday(&end, NULL);

    //device mem write back to host
    #pragma offload target(mic) \
    nocopy(buffer:length(n) alloc_if(0) free_if(1)) \
    out(data:length(n) alloc_if(0) free_if(1))
    {};

    return diffTime(end, start);
}

double tbb_sort(unsigned *data, size_t num) {
    task_scheduler_init init;
    struct timeval start, end;

    #pragma offload target(mic) inout(data:length(num) alloc_if(1) free_if(1)) 
    {
        gettimeofday(&start, NULL);
        parallel_sort(data, data + num);
        gettimeofday(&end, NULL);
    }
    return diffTime(end, start);
}

void testRadixSort(unsigned *arr, int len) {

    double myTime = radixSort(arr, len);

    //checking
    bool res = true;
    for(int i = 0;i < len-1; i++) {
       if (arr[i] > arr[i+1])  {
            res = false;
            break;
       } 
    }
    printRes("radix sort", res, myTime);
}

void testRadixSort_tbb(unsigned *arr_tbb, int len) {

    double myTime_tbb = tbb_sort(arr_tbb, len);

    //checking
    bool res_tbb = true;
    for(int i = 0;i < len-1; i++) {
       if (arr_tbb[i] > arr_tbb[i+1])   {
            res_tbb = false;
            break;
       }
    }
    printRes("tbb sort", res_tbb, myTime_tbb);
}