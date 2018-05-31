/*
 * build:
 * 1. set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. compile the file using:
 *      icc -O3 -o scanOMP scanOMP.cpp -fopenmp
 */
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#define MAX_THREAD_NUM (256)

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

//------------------------------------------------------------------
bool scanOMP(int len, double &totalTime) {
    totalTime = 0;
    bool res = true;
    int *input = new int[len];
    int *output = new int[len];
    for(int i = 0; i < len; i++) input[i] = 1;

    int experTime = 10;
    int normalCount = 0;
    double normalTempTime;
    bool next_valid = true;

    for(int e = 0; e < experTime; e++) {
        struct timeval start, end;
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
        double tempTime =  diffTime(end, start);

        if(e==0) {      //check
            for(int i = 0; i < len; i++) {
                if (output[i] != i+1) {
                    res = false;
                    break;
                }
            }
            normalTempTime = tempTime;
            totalTime = tempTime;
        }
        else if (res == true) {
            if (next_valid == true) {
                if (tempTime < normalTempTime*1.05) {      //with 5% error
                    if (tempTime*1.05 < normalTempTime) { //means the normalTempTime is an outlier
                        normalCount = 1;
                        normalTempTime = tempTime;
                        totalTime = tempTime;
                    }
                    else {  //temp time is correct
                        totalTime += tempTime;
                        normalCount++;
                    }
                }
                else next_valid = false; //tempTime is larger than the normal time, discard this and the next results
            }
            else    next_valid = true;

        }
        else {
            break;
        }
    }
    totalTime/= normalCount;

    delete[] input;
    delete[] output;
    return res;
}

int main()
{
    for(int scale = 10; scale <= 30; scale++) {
        int length = 1<<scale;

        std::cout<<scale<<" length: "<<length<<'\t';

        double totalTime;
        bool res = scanOMP(length, totalTime);
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
