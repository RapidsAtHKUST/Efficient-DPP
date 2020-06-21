/*
 * Execute on CPU:
 * 1. set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. compile the file using:
 *      icc -O3 -o mul_omp_cpu mul_omp.cpp -fopenmp
 * 3. Execute:
 *      ./mul_omp_cpu
 * To enable streaming store, modify the main function
 *
 * Execute on MIC (only native execution mode):
 * 1. Compile the file:
 *      icc -mmic -O3 -o mul_omp_mic mul_omp.cpp -fopenmp
 * 1.5 Compile with Streaming Store:
 *      icc -mmic -O3 -o mul_omp_mic_ss mul_omp.cpp -fopenmp -qopt-streaming-stores always
 * 2. Copy the executable file to MIC:
 *      scp mul_omp_mic mic0:~
 * 3. (optional) If the MIC does not have libiomp5.so, copy the library from .../intel/lib/mic to MIC:
 *      e.g.: scp libiomp5.so mic0:~
 * 4. (optional) Set the library path on MIC:
 *      e.g.: export LD_LIBRARY_PATH=~
 * 5. Execute:
 *      ./mul_omp_mic
 */
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <stdio.h>
#include <vector>
#define SCALAR (3)
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
double scale_omp(int *input, int *output, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < len; i++) {
        output[i] = input[i] * SCALAR;
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

double mem_access_omp_read(int *input, int *output, int len, int chunk_size) {

    struct timeval start, end;

    int num_threads;

//    #pragma omp parallel for schedule(static)
//    for(int i = 0; i < len; i++) {
//        input[i] = 1;
//    }
//    omp_set_dynamic(0);
//    omp_set_num_threads(10);

    gettimeofday(&start, NULL);
#pragma omp parallel
{
//    int thread_num = omp_get_thread_num();
//    int cpu_num = sched_getcpu();
//    printf("Thread %d on core %d\n",thread_num, cpu_num);

    long acc = 0;
    #pragma omp for schedule(dynamic,chunk_size)
    for(int i = 0; i < len; i++) {
        acc += input[i];
    }
    output[0] = acc;
}
    gettimeofday(&end, NULL);

    double tempTime =  diffTime(end, start);


    return tempTime;
}

//scalar multiplication with nontemporal streaming store
double scale_omp_ss(int *input, int *output, int len) {
#define SIMD_WIDTH  (8)
    struct timeval start, end;

    __m256i v = _mm256_set_epi32(SCALAR,SCALAR,SCALAR,SCALAR,SCALAR,SCALAR,SCALAR,SCALAR);

    gettimeofday(&start, NULL);
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len/8; i++) {
        register __m256i *dest = (__m256i*)output + i;
        register __m256i source = *((__m256i*)input + i);
        source = _mm256_mullo_epi32 (source, v);
        _mm256_stream_si256(dest,source);   //streaming store
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

void test_omp() {
    int len = 512 * 8192 * 100;  //1600MB
    std::cout<<"Data size(Multiplication): "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    int *input = new int[len];
    int *output = new int[len];
    for(int i = 0; i < len; i++) input[i] = i;

    int experTime = 30;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = scale_omp(input, output, len);
    }
    double aveTime = average_Hampel(times, experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<2*1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input)  delete[] input;
    if(output)  delete[] output;
}

void test_omp_read(int chunk_size) {
    int len = 512 * 8192 * 100;  //1600MB
    std::cout<<"Data size: "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    int *input = new int[len];
    int *output = new int[len];
    for(int i = 0; i < len; i++) input[i] = i;

    int experTime = 30;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = mem_access_omp_read(input, output, len, chunk_size);
    }
    double aveTime = average_Hampel(times, experTime);
    std::cout<<"Chunk size:"<<chunk_size<<"\tTime:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input)  delete[] input;
    if(output)  delete[] output;
}

void test_omp_ss() {
    int len = 512 * 8192 * 100;  //1600MB
    std::cout<<"Data size(Multiplication): "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    //use _mm_malloc
    int *input = (int*)_mm_malloc(sizeof(int)*len,32);
    int *output = (int*)_mm_malloc(sizeof(int)*len,32);
    for(int i = 0; i < len; i++) input[i] = i;

    int experTime = 150;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = scale_omp_ss(input, output, len);
    }
    double aveTime = average_Hampel(times, experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<2*1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input) _mm_free(input);
    if(output) _mm_free(output);
}

int main()
{
//    test_omp();
    for(int i = 64; i <= 524288; i<<=1)
        test_omp_read(i);

    //streaming store, only used on CPU
//    test_omp_ss();
    return 0;
}