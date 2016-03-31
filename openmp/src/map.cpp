#include "functions.h"
#include <immintrin.h>
#include <offload.h>

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

using namespace std;

//declare simd will make it faster from 188ms to 122ms
#pragma omp declare simd
__attribute__ ((target(mic))) int floorOfPower2(int a) {
	int k = 1;
	while (k < a) k <<=1;
	return k>>1;
}


double map(float *source, float *dest, int n /*, FuncType func*/) {
	
    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1))  
    {
    	gettimeofday(&start, NULL);
		#pragma omp parallel for simd
		for(int i = 0; i < n; i++) {
			dest[i] = floorOfPower2(source[i]);
			__m512 vec0, vec1, vec2;
			vec0 = _mm512_load_ps((float*)source + i);
			vec1 = _mm512_load_ps((float*)fixed);
			vec2 = _mm512_sub_ps(vec0, vec1);
			// _mm512_packstorelo_ps((void*)dest+i, vec2);
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	return diffTime(end, start);
}
