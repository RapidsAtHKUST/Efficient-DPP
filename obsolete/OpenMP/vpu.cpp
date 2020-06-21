#include "functions.h"

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

using namespace std;

#define FACTOR      (1.0001)
// s = s * con + con = FACTOR * s + FACTOR
#define MADD1_OP  s = _mm512_fmadd233_ps(s, con);

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP

double vpu_test(float *data, int n, int repeatTime){

    struct timeval start, end;

    #pragma offload target(mic)	\
    in(data:length(n) alloc_if(1) free_if(0))
    {}

    gettimeofday(&start, NULL);
    		
    #pragma offload target(mic)	\
    nocopy(data:length(n) alloc_if(0) free_if(0))
    {
    	#pragma omp parallel for schedule(auto) 
	    for (int gid = 0; gid < n/16; gid++)
	    {
	       	__declspec(target(mic)) register __m512 s = _mm512_load_ps(&data[gid*16]);
	       	__declspec(target(mic)) register __m512 con = _mm512_set1_ps(FACTOR);

	        for (int j=0 ; j<repeatTime ; ++j)
	        {
	            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
	            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
	            MADD1_MOP20 MADD1_MOP20
	        }
            
	        _mm512_store_ps((void*)(data+gid*16), s);
	    }
    }
    gettimeofday(&end, NULL);

    #pragma offload target(mic)	\
    out(data:length(n) free_if(1)) 
    {}

	return diffTime(end, start);
}