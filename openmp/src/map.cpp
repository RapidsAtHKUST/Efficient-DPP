#include "functions.h"
#include <immintrin.h>
#include <offload.h>

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

using namespace std;

//declare simd will make it faster from 188ms to 122ms
//change to intrinsic takes the same time as individual function
#pragma omp declare simd
__attribute__ ((target(mic))) int floorOfPower2(int v) {
	v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v>>1;
}


double map(int *source, int *dest, int n /*, FuncType func*/) {
	
    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(0)) \
    in(dest:length(n) alloc_if(1) free_if(0))  
    {}


    #pragma offload target(mic) \
    nocopy(source:length(n)) \
    nocopy(dest:length(n))  
    {
    	gettimeofday(&start, NULL);

    	//all 1's
    	__m512i constant_1 = _mm512_set1_epi32(1);
    	__m512i constant_2 = _mm512_set1_epi32(2);
    	__m512i constant_4 = _mm512_set1_epi32(4);
    	__m512i constant_8 = _mm512_set1_epi32(8);
    	__m512i constant_16 = _mm512_set1_epi32(16);

		#pragma omp parallel for simd
		for(int i = 0; i < n; i++) {
			dest[i] = floorOfPower2(source[i]);
			// __m512i vec0, vec1;
			// vec0 = _mm512_load_epi32(source + i);
			// vec0 = _mm512_sub_epi32(vec0, constant_1);

			// //>>1
			// vec1 = _mm512_srlv_epi32(vec0,constant_1);
			// vec0 = _mm512_or_epi32(vec0, vec1);

			// //>>2
			// vec1 = _mm512_srlv_epi32(vec0,constant_2);
			// vec0 = _mm512_or_epi32(vec0, vec1);

			// //>>4
			// vec1 = _mm512_srlv_epi32(vec0,constant_4);
			// vec0 = _mm512_or_epi32(vec0, vec1);

			// //>>8
			// vec1 = _mm512_srlv_epi32(vec0,constant_8);
			// vec0 = _mm512_or_epi32(vec0, vec1);

			// //>>16
			// vec1 = _mm512_srlv_epi32(vec0,constant_16);
			// vec0 = _mm512_or_epi32(vec0, vec1);

			// vec0 = _mm512_add_epi32(vec0, constant_1);
			// vec0 = _mm512_srlv_epi32(vec0,constant_1);

			// _mm512_store_epi32((void*)(dest+i), vec0);
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	#pragma offload target(mic) \
    out(source:length(n) alloc_if(0) free_if(1)) \
    out(dest:length(n) alloc_if(0) free_if(1))  
    {}

	return diffTime(end, start);
}
