#include "functions.h"

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

#define MAX_THREAD_NUM		(256)

using namespace std;

double mem_read_test(int *data, int n){

    struct timeval start, end;

    //just for storing the result for not optimizing out the read operations
    int *tempRes = new int[MAX_THREAD_NUM];
    memset(tempRes,sizeof(int)*MAX_THREAD_NUM,0);

    #pragma offload target(mic)	                \
    in(data:length(n) alloc_if(1) free_if(0) )	\
    inout(tempRes:length(MAX_THREAD_NUM) alloc_if(1) free_if(0) )
    {}

    gettimeofday(&start, NULL);

    #pragma offload target(mic)	                    \
    nocopy(data:length(n) alloc_if(0) free_if(0) )	\
    nocopy(tempRes:length(MAX_THREAD_NUM) alloc_if(0) free_if(0) )
    {
    	#pragma omp parallel 
    	{
    		int tid = omp_get_thread_num();
    		int privateSum = 0;	//each thread reads numbers and add to it
    		#pragma omp for schedule(auto) 
		    for (int idx = 0; idx < n; idx++) {
		       	privateSum += data[idx];
		    }
		    tempRes[tid] = privateSum;
    	}
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic)	                    \
    out(data:length(n) free_if(1))	                \
    out(tempRes:length(MAX_THREAD_NUM) free_if(1)) 
    {}

    delete[] tempRes;

	return diffTime(end, start);
}

double mem_write_test(int *data, int n){

    struct timeval start, end;

    #pragma offload target(mic)	\
    in(data:length(n) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
    		
    #pragma offload target(mic)	\
    nocopy(data:length(n) alloc_if(0) free_if(0))
    {
    	#pragma omp parallel for schedule(auto) 
	    for (int idx = 0; idx < n; idx++) {
	       	data[idx] = idx;
	    }
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic)	\
    out(data:length(n) free_if(1))
    {}

	return diffTime(end, start);
}

double mem_mul_test(int *input, int*output, int n){


    struct timeval start, end;

    #pragma offload target(mic) \
    in(input:length(n) alloc_if(1) free_if(0)) \
    inout(output:length(n) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
            
    #pragma offload target(mic) \
    nocopy(input:length(n) alloc_if(0) free_if(0))  \
    nocopy(output:length(n) alloc_if(0) free_if(0))
    {
        // __assume_aligned(input, 64);
        // __assume_aligned(output, 64);

        #pragma omp parallel for schedule(auto) 
        for (int idx = 0; idx < n; idx++) {
            output[idx] = input[idx] * 3.0f;
        }
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic) \
    out(input:length(n) free_if(1)) \
    out(output:length(n) free_if(1))
    {}

    return diffTime(end, start);
}

double mem_add_test(int *input,int *input_2,  int*output, int n){

    struct timeval start, end;

    #pragma offload target(mic) \
    in(input:length(n) alloc_if(1) free_if(0)) \
    in(input_2:length(n) alloc_if(1) free_if(0)) \
    inout(output:length(n) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
            
    #pragma offload target(mic) \
    nocopy(input:length(n) alloc_if(0) free_if(0))  \
    nocopy(input_2:length(n) alloc_if(0) free_if(0))  \
    nocopy(output:length(n) alloc_if(0) free_if(0))
    {
        // __assume_aligned(input, 64);
        // __assume_aligned(output, 64);

        #pragma omp parallel for schedule(auto) 
        for (int idx = 0; idx < n; idx++) {
            output[idx] = input[idx] + input_2[idx];
        }
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic) \
    out(input:length(n) free_if(1)) \
    out(input_2:length(n) free_if(1)) \
    out(output:length(n) free_if(1))
    {}

    return diffTime(end, start);
}