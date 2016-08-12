#include "functions.h"

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

#define MAX_THREAD_NUM		(256)

using namespace std;

double mem_read_test(float *data, int n, int repeatTime){

    struct timeval start, end;

    //just for storing the result for not optimizing out the read operations
    float *tempRes = new float[MAX_THREAD_NUM];
    memset(tempRes,sizeof(float)*MAX_THREAD_NUM,0);

    #pragma offload target(mic)	\
    in(data:length(n) alloc_if(1) free_if(0))	\
    inout(tempRes:length(MAX_THREAD_NUM) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
    		
    #pragma offload target(mic)	\
    nocopy(data:length(n) alloc_if(0) free_if(0))	\
    nocopy(tempRes:length(MAX_THREAD_NUM) alloc_if(0) free_if(0))
    {
    	#pragma omp parallel 
    	{
    		int tid = omp_get_thread_num();
    		for(int i = 0; i < repeatTime; i++) {
	    		float privateSum = 0.0;			//each thread reads numbers and add to it
	    		#pragma omp for schedule(auto) 
			    for (int idx = 0; idx < n; idx++) {
			       	privateSum += data[idx];
			    }
			    tempRes[tid] += privateSum;
    		}
    	}
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic)	\
    out(data:length(n) free_if(1))	\
    out(tempRes:length(MAX_THREAD_NUM) free_if(1)) 
    {}

    delete[] tempRes;

	return diffTime(end, start);
}

double mem_write_test(float *data, int n, int repeatTime){

    struct timeval start, end;

    #pragma offload target(mic)	\
    in(data:length(n) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
    		
    #pragma offload target(mic)	\
    nocopy(data:length(n) alloc_if(0) free_if(0))
    {
		// float write = 2.8913;
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