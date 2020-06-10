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
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <stdio.h>
#include <vector>
using namespace std;

bool pair_cmp (pair<double, double> i , pair<double, double> j) {
    return i.first < j.first;
}

double average_Hampel(double *input, int num) {
    int valid = 0;
    double total = 0;

    double *temp_input = new double[num];
    vector< pair<double, double> > myabs_and_input_list;

    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++)
        myabs_and_input_list.push_back(make_pair(fabs(temp_input[i]-mean),temp_input[i]));

    typedef vector< pair<double, double> >::iterator VectorIterator;
    sort(myabs_and_input_list.begin(), myabs_and_input_list.end(), pair_cmp);

    if (num % 2 == 0)  abs_mean = 0.5*(myabs_and_input_list[num/2-1].first + myabs_and_input_list[num/2].first);
    else               abs_mean = myabs_and_input_list[(num-1)/2].first;

    abs_mean /= 0.6745;

    for(VectorIterator iter = myabs_and_input_list.begin(); iter != myabs_and_input_list.end(); iter++) {
        if (abs_mean == 0) { /*if abs_mean=0,only choose those with abs=0*/
            if (iter->first == 0) {
                total += iter->second;
                valid ++;
            }
        }
        else {
            double div = iter->first / abs_mean;
            if (div <= 3.5) {
                total += iter->second;
                valid ++;
            }
        }
    }
    total = 1.0 * total / valid;
    if(temp_input)  delete[] temp_input;
    return total;
}

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

//scalar multiplication
double mem_access_omp(int *input, int *output, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < len; i++) {
        output[i] = input[i];
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

//memory copy with nontemporal streaming store
double mem_access_omp_ss(int *input, int *output, int len) {
#define SIMD_WIDTH  (8)
    struct timeval start, end;

    gettimeofday(&start, NULL);
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len/8; i++) {
        register __m256i *dest = (__m256i*)output + i;
        register __m256i source = *((__m256i*)input + i);
        _mm256_stream_si256(dest,source);   //streaming store
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

void test_omp(int len) {
    std::cout<<"Data size(Copy): "<<len<<" ("<<1.0*len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    int *input = new int[len];
    int *output = new int[len];
    for(int i = 0; i < len; i++) input[i] = i;

    int experTime = 150;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = mem_access_omp(input, output, len);
    }

    double aveTime = average_Hampel(times, experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len/1024/1024/1024/aveTime*1e3<<" GKeys/s"<<std::endl; //compared with scan

    if(input)  delete[] input;
    if(output)  delete[] output;
}

void test_omp_ss(int len) {
    std::cout<<"Data size(Copy): "<<len<<" ("<<1.0*len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    //use _mm_malloc
    int *input = (int*)_mm_malloc(sizeof(int)*len,32);
    int *output = (int*)_mm_malloc(sizeof(int)*len,32);
    for(int i = 0; i < len; i++) input[i] = i;

    int experTime = 150;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = mem_access_omp_ss(input, output, len);
    }
    double aveTime = average_Hampel(times, experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len/1024/1024/1024/aveTime*1e3<<" GKey/s"<<std::endl; //compared with scan

    if(input) _mm_free(input);
    if(output) _mm_free(output);
}

int main()
{
//    test_omp();

    for(int scale = 10; scale <= 30; scale++) {
        unsigned data_size = 1<<scale;
//        test_omp_ss(data_size);
        test_omp(data_size);
    }

    return 0;
}