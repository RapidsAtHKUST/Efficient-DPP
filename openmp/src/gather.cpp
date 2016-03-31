#include "functions.h"
#include <immintrin.h>
using namespace std;

double gather(int *source, int *dest, int n, int *loc) {
    kmp_set_defaults("KMP_AFFINITY=compact");
    struct timeval start, end;

    #pragma offload target(mic) 				\
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1)) 	\
    in(loc:length(n) alloc_if(1) free_if(1))
    {
    	gettimeofday(&start, NULL);

		#pragma omp parallel for simd
		for(int i = 0; i < n; i++) {
			dest[i] = source[loc[i]];
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}
	return diffTime(end,start);
}

double gather_intr(int *source, int *dest, int len, int *loc) {
    kmp_set_defaults("KMP_AFFINITY=compact");
	struct timeval start, end;
	
	#pragma offload target(mic) 					\
	out(dest:length(len) alloc_if(1) free_if(1))	\
	in(source:length(len) alloc_if(1) free_if(1))	\
	in(loc:length(len) alloc_if(1) free_if(1))
	{
		gettimeofday(&start, NULL);
		#pragma omp parallel for 
		for(int i = 0; i < len; i += 16) {
			__m512i index = _mm512_load_epi32(loc + i);
			__m512 res = _mm512_i32gather_ps(index, source, sizeof(int));
			_mm512_store_ps(dest + i , res);
		}
		#pragma omp barrier
		gettimeofday(&end, NULL);
	}
	return diffTime(end, start);
}

void testGather(int *source, int *dest, int *loc, int n) {

	double myTime = gather(source, dest, n, loc);

	//checking
	bool res = true;
	for(int i=  0; i < n; i++) {
		if (dest[i] != source[loc[i]])	{
			res = false;
			break;
		}
	}
	cout<<"Num: "<<n<<endl;
	cout<<"Time: "<<myTime<<" ms."<<endl;
	if (res)	cout<<"Right!"<<endl;
	else		cout<<"Wrong!"<<endl;
}

void testGather_intr(int *source, int *dest, int *loc, int n) {

	double myTime = gather_intr(source, dest, n, loc);

	//checking
	bool res = true;
	for(int i=  0; i < n; i++) {
		if (dest[i] != source[loc[i]])	{
			res = false;
			break;
		}
	}
	cout<<"Num: "<<n<<endl;
	cout<<"Time: "<<myTime<<" ms."<<endl;
	if (res)	cout<<"Right!"<<endl;
	else		cout<<"Wrong!"<<endl;
}