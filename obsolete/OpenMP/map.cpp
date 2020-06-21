#include "functions.h"
#include <immintrin.h>
#include <offload.h>

#pragma offload_attribute(push, target(mic))
#include <immintrin.h>
#pragma offload_attribute(pop)

using namespace std;

//declare simd will make it faster from 188ms to 122ms
//change to intrinsic takes the same time as individual function
//individual function has no vectorization!!

// #pragma omp declare simd
// __attribute__ ((target(mic))) int floorOfPower2(int v) {
// 	v--;
//     v |= v >> 1;
//     v |= v >> 2;
//     v |= v >> 4;
//     v |= v >> 8;
//     v |= v >> 16;
//     v++;
//     return v>>1;
// }

double map_MIC(int *source, int *dest, int n, int k) {
	
    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) free_if(0)) \
    nocopy(dest:length(n) alloc_if(1) free_if(0))  
    {}	

    #pragma offload target(mic) \
    nocopy(source:length(n) alloc_if(0) free_if(0)) \
    nocopy(dest:length(n) alloc_if(0) free_if(0))  
    {	
		gettimeofday(&start, NULL);
    	#pragma omp parallel 
    	{
			#pragma omp for schedule(auto) nowait
			for(int i = 0; i < n; i++) {
				float pi = 0;
				int v = source[i];
				for(int ki = 0; ki <= k; ki++){
					pi += (ki+v)/(2*k*0.29+1.33);
				}
				pi *= 4;
				dest[i] = (int)pi;
			}
		}
		gettimeofday(&end, NULL);
	}

	#pragma offload target(mic) \
    out(dest:length(n))  
    {}

	return diffTime(end, start);
}

double map_CPU(int *source, int *dest, int n) {
	
    struct timeval start, end;

	gettimeofday(&start, NULL);

#pragma omp parallel for schedule(auto)
	for(int i = 0; i < n; i++) {
		dest[i] = source[i] + 1;
	}
	gettimeofday(&end, NULL);
			
	return diffTime(end, start);
}

// float pi = 0;
				// for(int k = 0; k <= 250; k++){
				// 	// if (k % 2 == 0)	pi += (k+v)/(2*v+1);
				// 	// else 			pi -= (k+v)/(2*v+1);

				// 	pi += (k+v)/(2*v+1);
				// }
				// pi *= 4;
				// dest[i] = (int)pi;
				// v--;
			 //    v |= v >> 1;
			 //    v |= v >> 2;
			 //    v |= v >> 4;
			 //    v |= v >> 8;
			 //    v |= v >> 16;
			 //    v++;
			 //    v>>=1;
			 //    dest[i] = v;
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