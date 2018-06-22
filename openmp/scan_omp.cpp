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
#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <omp.h>
#define MAX_THREAD_NUM (256)

double averageHampel(double *input, int num) {
    int valid = 0;
    double total = 0;

    double *temp_input = new double[num];
    double *myabs = new double[num];
    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    std::sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++)    myabs[i] = fabs(temp_input[i]-mean);

    std::sort(myabs, myabs+num);
    if (num % 2 == 0)  abs_mean = 0.5*(myabs[num/2-1] + myabs[num/2]);
    else               abs_mean = myabs[(num-1)/2];

    abs_mean /= 0.6745;

    for(int i = 0; i < num; i++) {
        double div = myabs[i] / abs_mean;
        if (div <= 3.5) {
            total += temp_input[i];
            valid ++;
        }
    }
    total = 1.0 * total / valid;

    if(temp_input)  delete[] temp_input;
    if (myabs)      delete[] myabs;
    return total;
}

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

//reduce-sca-scan scheme
double scan_omp(int *input, int* output, int len) {
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
        for (int i = 0; i < len; i++) {
            localSum += input[i];
            output[i] = localSum;
        }
        reduceSum[tid] = localSum;

#pragma omp barrier

        //exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
#pragma omp single
        {
            int temp = 0;
            for (int i = 0; i < nthreads; i++) {
                reduceSum_scanned[i] = temp;
                temp = temp + reduceSum[i];
            }
        }

        //scatter back
#pragma omp for schedule(static) nowait
        for (int i = 0; i < len; i++) {
            output[i] += reduceSum_scanned[tid];
        }
    }
    gettimeofday(&end, NULL);
    return diffTime(end, start);
}

void test_scan_omp() {
    for(int scale = 10; scale <= 30; scale++) {
        bool res = true;
        int length = 1<<scale;
        std::cout<<scale<<" length: "<<length<<'\t';

        int *input = new int[length];
        int *output = new int[length];
        for(int i = 0; i < length; i++) input[i] = 1;

        int experTime = 10;
        double tempTimes[experTime];
        for(int e = 0; e < experTime; e++) {
            tempTimes[e] = scan_omp(input, output, length);
            if (e == 0) {         //check
                for (int i = 0; i < length; i++) {
                    if (output[i] != i + 1) {
                        res = false;
                        break;
                    }
                }
            }
        }
        double aveTime = averageHampel(tempTimes,experTime);

        if(input)   delete[] input;
        if(output)  delete[] output;

        if (res)
            std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
                     <<"Throughput:"<<1.0*length* sizeof(int)/1024/1024/1024/aveTime*1e3/sizeof(int)<<" Gkeys/s"<<std::endl;
        else std::cout<<"wrong results"<<std::endl;
    }
}

int main() {
    test_scan_omp();
    return 0;
}