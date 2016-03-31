#include "functions.h"
using namespace std;

double scatter(int *source, int *dest, int n, int *loc) {
    kmp_set_defaults("KMP_AFFINITY=compact");
    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1)) \
    in(loc:length(n) alloc_if(1) free_if(1))
    {
    	gettimeofday(&start, NULL);
		#pragma omp parallel for simd
		for(int i = 0; i < n; i++) {
			dest[loc[i]] = source[i];
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	return diffTime(end,start);
}

double scatter_intr(int *source, int *dest, int n, int *loc) {
    kmp_set_defaults("KMP_AFFINITY=compact");
    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1)) \
    in(loc:length(n) alloc_if(1) free_if(1))
    {
    	gettimeofday(&start, NULL);
		#pragma omp parallel for simd
		for(int i = 0; i < n; i+=16) {
			__m512i index = _mm512_load_epi32(loc + i);
			__m512 source_piece = _mm512_load_ps(source + i);
			_mm512_i32scatter_ps(dest, index, source_piece, sizeof(int));
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	return diffTime(end,start);
}

void testScatter(int *source, int *dest, int *loc, int n) {

	double myTime = scatter(source, dest, n, loc);

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

void testScatter_intr(int *source, int *dest, int *loc, int n) {

	double myTime = scatter_intr(source, dest, n, loc);

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