#include "functions.h"
using namespace std;

#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE-1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)

double radixSort(size_t n, unsigned *data) {
    
    unsigned *buffer;
    int total_digits = sizeof(unsigned)*8;
 
    kmp_set_defaults("KMP_AFFINITY=compact");

    struct timeval start, end;

    //device memory allocation
    #pragma offload target(mic) \
    nocopy(buffer:length(n) alloc_if(1) free_if(0)) \
    in(data:length(n) alloc_if(1) free_if(0))
    {};

    gettimeofday(&start, NULL);
    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {

        size_t bucket[BASE] = {0};
        #pragma offload target(mic) nocopy(data) nocopy(buffer)
        {
            //done by all the threads
            #pragma omp parallel
            {
                size_t local_bucket[BASE] = {0};    //local array
                #pragma omp for schedule(static) nowait
                for(int i = 0; i < n; i++){
                    local_bucket[DIGITS(data[i], shift)]++;
                }
                #pragma omp critical
                for(int i = 0; i < BASE; i++) {
                    bucket[i] += local_bucket[i];
                }
                #pragma omp barrier
                #pragma omp single
                for (int i = 1; i < BASE; i++) {
                    bucket[i] += bucket[i - 1];
                }
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
                    if(cur_t == tid) { 
                        for(int i = 0; i < BASE; i++) {
                            bucket[i] -= local_bucket[i];
                            local_bucket[i] = bucket[i];
                        }
                    } else { //just do barrier
                        #pragma omp barrier
                    }
                }
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
    out(data:length(n) alloc_if(0) free_if(1)) \
    nocopy(buffer:length(n) free_if(1) alloc_if(0))
    {};

    return diffTime(end,start);
}

void testRadixSort() {
    int len = 16000000;
    unsigned *arr = new unsigned[len];

    for(int i = 0;i < len; i++) {
        arr[i] = rand() % INT_MAX;
        // cout<<arr[i]<<' ';
    }
    // cout<<endl;
    double myTime = radixSort(len, arr);

    //checking
    bool res = true;
    for(int i = 0;i < len-1; i++) {
        // cout<<arr[i]<<' ';
       if (arr[i] > arr[i+1])   res = false;
    }
    // cout<<endl;
    if (res)    cout<<"Right!"<<endl;
    else        cout<<"Wrong!"<<endl;

    cout<<"Time: "<<myTime<<" ms."<<endl;
    delete[] arr;
}