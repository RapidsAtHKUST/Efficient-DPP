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
#define SCALAR (3)

double averageHampel(double *input, int num) {
    int valid = 0;
    float total = 0;

    double *temp_input = new double[num];
    double *myabs = new double[num];
    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    std::sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++) myabs[i] = fabs(temp_input[i]-mean);

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

//scalar multiplication
double mem_access_omp(int *input, int *output, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
    #pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++) {
        output[i] = input[i] * SCALAR;
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

//scalar multiplication with nontemporal streaming store
double mem_access_omp_ss(int *input, int *output, int len) {
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

    int experTime = 150;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = mem_access_omp(input, output, len);
    }
    double aveTime = averageHampel(times,experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<2*1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

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
        times[e] = mem_access_omp_ss(input, output, len);
    }
    double aveTime = averageHampel(times,experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<2*1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input) _mm_free(input);
    if(output) _mm_free(output);
}

int main()
{
    test_omp();

    //streaming store, only used on CPU
//    test_omp_ss();
    return 0;
}