//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#ifdef __JETBRAINS_IDE__
#include "cuda_fake/fake.h"
#include "openmp_fake.h"
#endif

#include <iostream>
#include <omp.h>
#include "log.h"

double compute_bandwidth(uint64_t num, int wordSize, double kernel_time) {
    return 1.0*num/1024/1024/1024*wordSize/kernel_time * 1000 ;
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