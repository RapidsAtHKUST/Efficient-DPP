/*
 * build:
 * 1. set the host env variable by using:
 *      source /usr/local/intel/bin/compilervars.sh intel64 (setting the MIC_LD_LIBRARY and MIC_LIBRARY envs)
 * 2. compile the file using:
 *      icc -O3 -o scanOMP_mic scanOMP_mic.cpp -fopenmp
 */
#pragma offload_attribute(push, target(mic))
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#pragma offload_attribute(pop)

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

//------------------------------------------------------------------
#define MAX_THREAD_NUM (256)
bool scanOMP_mic(int len, double &totalTime) {
    totalTime = 0;
    bool res = true;
    int *input = (int*)_mm_malloc(sizeof(int)*len, 64);
    int *output = (int*)_mm_malloc(sizeof(int)*len, 64);
    for(int i = 0; i < len; i++) input[i] = 1;

    int experTime = 10;
    int normalCount = 0;
    double normalTempTime;

    for(int e = 0; e < experTime; e++) {
        struct timeval start, end;

        #pragma offload target(mic) \
        in(input:length(len) alloc_if(1) free_if(0)) \
        in(output:length(len) alloc_if(1) free_if(0))
        {}

        #pragma offload target(mic) \
        nocopy(input:length(len) ) \
        nocopy(output:length(len) )
        {
            int reduceSum[MAX_THREAD_NUM] = {0};
            int reduceSum_scanned[MAX_THREAD_NUM] = {0};
            gettimeofday(&start, NULL);

            #pragma omp parallel
            {
                int nthreads = omp_get_num_threads();
                int tid = omp_get_thread_num();

                int localSum = 0;

                //reduce & local prefix sum
                #pragma omp for schedule(static) nowait
                for(int i = 0; i < len; i++) {
                    localSum += input[i];
                    output[i] = localSum;
                }
                reduceSum[tid] = localSum;

                #pragma omp barrier

                //exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
                #pragma omp single
                {
                    int temp = 0;
                    for(int i = 0; i < nthreads; i++) {
                        reduceSum_scanned[i] = temp;
                        temp = temp + reduceSum[i];
                    }
                }

                //scatter back
                #pragma omp for schedule(static) nowait
                for(int i = 0; i < len; i++) {
                    output[i] += reduceSum_scanned[tid];
                }
            }
            gettimeofday(&end, NULL);
        }

        #pragma offload target(mic) \
        out(input:length(len) alloc_if(0) free_if(1)) \
        out(output:length(len) alloc_if(0) free_if(1))
        {}

        double tempTime =  diffTime(end, start);

        if(e==0) {      //check
            for(int i = 0; i < len; i++) {
                if (output[i] != i+1) {
                    res = false;
                    break;
                }
            }
            //normalTempTime = tempTime;
            //totalTime = tempTime;
        }
        else if (res == true) {
            // if (tempTime < normalTempTime*1.05) {      //with 5% error
            //     if (tempTime*1.05 < normalTempTime) { //means the normalTempTime is an outlier
            //         normalCount = 1;
            //         normalTempTime = tempTime;
            //         totalTime = tempTime;
            //     }
            //     else {  //temp time is correct
            //         totalTime += tempTime;
            //         normalCount++;
            //     }
            // }
            if (e >= experTime/2) {
                totalTime += tempTime;
                normalCount++;
            }
        }
        else {
            break;
        }
    }
    totalTime/= normalCount;

    _mm_free(input);
    _mm_free(output);
    return res;
}

int main()
{
    for(int scale = 10; scale <= 30; scale++) {
        int length = 1<<scale;

        std::cout<<scale<<" length: "<<length<<'\t';

        double totalTime;
        bool res = scanOMP_mic(length, totalTime);
        if (res) {
            std::cout<<"Time:"<<totalTime<<" ms"<<'\t'
                <<"Throughput:"<<1.0*length* sizeof(int)/1024/1024/1024/totalTime*1e3<<" GB/s"<<std::endl;
        }
        else {
            std::cout<<"wrong results"<<std::endl;
        }
    }
    return 0;
}
