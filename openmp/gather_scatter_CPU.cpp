/*
 * Execute on CPU:
 * 1. set the library path:
 *      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries/linux/lib/intel64/
 * 2. compile the file using:
 *      icc -O3 -o gather_scatter_CPU gather_scatter_CPU.cpp -fopenmp
 * 3. Execute:
 *      ./gather_scatter_CPU
 * To enable streaming store, modify the main function
 *
 */
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <stdio.h>
#include <vector>
#define MAX_SHUFFLE_TIME (2099999999)
#define EXPER_TIME (50)

using namespace std;

bool pair_cmp (pair<double, double> i , pair<double, double> j) {
    return i.first < j.first;
}

double averageHampel(double *input, int num) {
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

//gather
double gather(int *input, int *output, int *idx, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
    #pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++) {
        output[i] = input[idx[i]]; /*with -O3, this inst is auto-vectorized*/
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

double scatter(int *input, int *output, int *idx, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len; i++) {
        output[idx[i]] = input[i];
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

//gather with streaming stores
double gather_intrinsic(int *input, int *output, int *idx, int len) {

    struct timeval start, end;

    gettimeofday(&start, NULL);
#pragma omp parallel for schedule(auto)
    for(int i = 0; i < len/8; i++) {
        register __m256i indexes = *((__m256i*)idx + i);
        register __m256i *dest = (__m256i*)output + i;
        register __m256i source = _mm256_i32gather_epi32(input, indexes, sizeof(int));
        _mm256_store_si256(dest, source);
    }
    gettimeofday(&end, NULL);
    double tempTime =  diffTime(end, start);

    return tempTime;
}

void test_gather(int len) {
    std::cout<<"Data size(Gather): "<<len<<" ("<<1.0*len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    int *input = new int[len];
    int *idx = new int[len];
    int *output = new int[len];
    srand((unsigned)time(NULL)); sleep(1);

    for(int i = 0; i < len; i++){
        input[i] = rand()%len;
        idx[i] = i;
    }

    /*shuffle the indexes*/
    unsigned shuffleTime = (len * 3 < MAX_SHUFFLE_TIME)? len*3 : MAX_SHUFFLE_TIME;

    /*data shuffling*/
    int temp, from = 0, to = 0;
    for(int i = 0; i < shuffleTime; i++) {
        from = rand() % len;
        to = rand() % len;
        temp = idx[from];
        idx[from] = idx[to];
        idx[to] = temp;
    }

    int experTime = EXPER_TIME;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = gather(input, output, idx, len);

        if (e == 0) { /*check the outputs*/
            bool res = true;
            for(int i = 0; i < len; i++) {
                if(output[i] != input[idx[i]]) {
                    res = false;
                    break;
                }
            }
            if (!res)   cout<<"Wrong result."<<endl;
        }
    }

    double aveTime = averageHampel(times,experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input)  delete[] input;
    if(output)  delete[] output;
    if(idx)  delete[] idx;
}

void test_scatter(int len) {
    std::cout<<"Data size(Scatter): "<<len<<" ("<<1.0*len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    int *input = new int[len];
    int *idx = new int[len];
    int *output = new int[len];
    srand((unsigned)time(NULL)); sleep(1);

    for(int i = 0; i < len; i++){
        input[i] = rand() % len;
        idx[i] = i;
    }

    /*shuffle the indexes*/
    unsigned shuffleTime = (len * 3 < MAX_SHUFFLE_TIME)? len*3 : MAX_SHUFFLE_TIME;

    /*data shuffling*/
    int temp, from = 0, to = 0;
    for(int i = 0; i < shuffleTime; i++) {
        from = rand() % len;
        to = rand() % len;
        temp = idx[from];
        idx[from] = idx[to];
        idx[to] = temp;
    }

    int experTime = EXPER_TIME;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = scatter(input, output, idx, len);

        if (e == 0) { /*check the outputs*/
            bool res = true;
            for(int i = 0; i < len; i++) {
                if(output[idx[i]] != input[i]) {
                    res = false;
                    break;
                }
            }
            if (!res)   cout<<"Wrong result."<<endl;
        }
    }

    double aveTime = averageHampel(times,experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input)  delete[] input;
    if(output)  delete[] output;
    if(idx)  delete[] idx;
}

void test_gather_intrinsic(int len) {
    std::cout<<"Data size(Gather): "<<len<<" ("<<1.0*len* sizeof(int)/1024/1024<<"MB)"<<'\t';

    //use _mm_malloc
    int *input = (int*)_mm_malloc(sizeof(int)*len,32);
    int *output = (int*)_mm_malloc(sizeof(int)*len,32);
    int *idx = (int*)_mm_malloc(sizeof(int)*len, 32);
    srand((unsigned)time(NULL)); sleep(1);

    for(int i = 0; i < len; i++){
        input[i] = rand() % len;
        idx[i] = i;
    }

    /*shuffle the indexes*/
    unsigned shuffleTime = (len * 3 < MAX_SHUFFLE_TIME)? len*3 : MAX_SHUFFLE_TIME;

    /*data shuffling*/
    int temp, from = 0, to = 0;
    for(int i = 0; i < shuffleTime; i++) {
        from = rand() % len;
        to = rand() % len;
        temp = idx[from];
        idx[from] = idx[to];
        idx[to] = temp;
    }

    int experTime = EXPER_TIME;
    double times[experTime];
    for(int e = 0; e < experTime; e++) {
        times[e] = gather_intrinsic(input, output, idx, len);

        if (e == 0) { /*check the outputs*/
            bool res = true;
            for(int i = 0; i < len; i++) {
                if(output[i] != input[idx[i]]) {
                    res = false;
                    break;
                }
            }
            if (!res)   cout<<"Wrong result."<<endl;
        }
    }
    double aveTime = averageHampel(times,experTime);
    std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
             <<"Throughput:"<<1.0*len* sizeof(int)/1024/1024/1024/aveTime*1e3<<" GB/s"<<std::endl;

    if(input) _mm_free(input);
    if(output) _mm_free(output);
    if(idx) _mm_free(idx);
}

int main()
{
    for(int data_size_MB = 128; data_size_MB < 4096; data_size_MB += 256) {
        int data_size = data_size_MB/ sizeof(int) * 1024 * 1024;
//        test_gather_intrinsic(data_size);
        test_scatter(data_size);
    }

    return 0;
}