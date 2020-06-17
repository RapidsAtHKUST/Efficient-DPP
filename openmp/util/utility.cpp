//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#ifdef __JETBRAINS_IDE__
#include "openmp_fake.h"
#endif

#include <iostream>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "log.h"
#include "utility.h"
using namespace std;

bool pair_cmp (pair<double, double> i , pair<double, double> j) {
    return i.first < j.first;
}

double compute_bandwidth(uint64_t num, int wordSize, double kernel_time) {
    return 1.0*num/1024/1024/1024*wordSize/kernel_time * 1000 ;
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

/*data generators*/
/*
 * Generate random uniform int value array
 * */
void random_generator_int(int *keys, uint64_t length, int max, unsigned long long seed) {
#pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int my_seed = seed + tid;
#pragma omp for schedule(dynamic)
        for(int i = 0; i < length ; i++)    keys[i] = rand_r(&my_seed) % max;
    }
}

/*
 * Generate random uniform unique int value array
 * */
void random_generator_int_unique(int *keys, uint64_t length) {
    srand((unsigned)time(nullptr));
#pragma omp parallel for
    for(int i = 0; i < length ; i++) {
        keys[i] = i;
    }
    log_trace("Key assignment finished");
    /*shuffling*/
    for(auto i = length-1; i > 0; i--) {
        auto from = rand() % i;
        auto to = rand() % i;
        std::swap(keys[from], keys[to]);
    }
    log_trace("Key shuffling finished");
}