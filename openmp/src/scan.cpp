// #pragma offload_attribute(push, target(mic))
#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/parallel_for.h"
#include <immintrin.h>
// #pragma offload_attribute(pop)

#include "functions.h"

using namespace std;
using namespace tbb;

#define MAX_THREAD_NUM	(256)

//tbb scan (inclusive) reach 80% utilization
template<typename T> class ScanBody_in {
 	T sum;
 	T* const y;
 	const T* const x; 
public:
 	ScanBody_in( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
 	T get_sum() const {return sum;}
 
 	template<typename Tag>
 	void operator()( const blocked_range<int>& r, Tag ) {
 		T temp = sum;
 		int end = r.end();
 		for( int i=r.begin(); i<end; ++i ) {
 			temp = temp + x[i];
 			if( Tag::is_final_scan() )
 				y[i] = temp;
 		}
 		sum = temp;
 	}
 	ScanBody_in( ScanBody_in& b, split ) : x(b.x), y(b.y), sum(0) {}
 	void reverse_join( ScanBody_in& a ) { sum = a.sum + sum;}
 	void assign( ScanBody_in& b ) {sum = b.sum;}
};

template<typename T> class ScanBody_ex {
 	T sum;
 	T* const y;
 	const T* const x; 
public:
 	ScanBody_ex( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
 	T get_sum() const {return sum;}
 
 	template<typename Tag>
 	void operator()( const blocked_range<int>& r, Tag ) {
 		T temp = sum;
 		int end = r.end();
 		for( int i=r.begin(); i<end; ++i ) {
 			if( Tag::is_final_scan() )
 				y[i] = temp;
 			temp = temp + x[i];
 			
 		}
 		sum = temp;
 	}
 	ScanBody_ex( ScanBody_ex& b, split ) : x(b.x), y(b.y), sum(0) {}
 	void reverse_join( ScanBody_ex& a ) { sum = a.sum + sum;}
 	void assign( ScanBody_ex& b ) {sum = b.sum;}
};

double scan_tbb_MIC(int *a, int *b, int n, int pattern) {
	// task_scheduler_init init;
	// struct timeval start, end;

	// if (pattern == 1) {
	// 	#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
	// 	{
	// 		gettimeofday(&start, NULL);
	// 		ScanBody_ex<int> body(b,a);
	//  		parallel_scan( blocked_range<int>(0,n), body );
	// 		gettimeofday(&end, NULL);
	// 	}
	// }
	// else {
	// 	#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
	// 	{
	// 		gettimeofday(&start, NULL);
	// 		ScanBody_in<int> body(b,a);
	//  		parallel_scan( blocked_range<int>(0,n), body );
	// 		gettimeofday(&end, NULL);
	// 	}
	// }
	// return diffTime(end, start);
}

double scan_tbb_CPU(int *a, int *b, int n, int pattern) {
	task_scheduler_init init;
	struct timeval start, end;

	if (pattern == 1) {
		gettimeofday(&start, NULL);
		ScanBody_ex<int> body(b,a);
 		parallel_scan( blocked_range<int>(0,n), body );
		gettimeofday(&end, NULL);
	}
	else {
		gettimeofday(&start, NULL);
		ScanBody_in<int> body(b,a);
 		parallel_scan( blocked_range<int>(0,n), body );
		gettimeofday(&end, NULL);
	}
	return diffTime(end, start);
}

//scan function using OpenMP on CPU
//pattern : 0 for inclusive, 1 for exclusive
double scan_omp_CPU(int *a, int *b, int n, int pattern) {
	struct timeval start, end;

	int reduceSum[MAX_THREAD_NUM] = {0};
	int reduceSum_scanned[MAX_THREAD_NUM] = {0};

	gettimeofday(&start, NULL);
	#pragma omp parallel 
	{
		int nthreads = omp_get_num_threads();
    	int tid = omp_get_thread_num();

    	int localSum = 0;

        //reduce & local prefix sum 
		#pragma omp for schedule(static) nowait 
		for(int i = 0; i < n; i++) {
			localSum += a[i];
			b[i] = localSum;
			}	
			reduceSum[tid] = localSum;
			
			#pragma omp barrier

			//exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
			#pragma omp single 
			{
 			int temp = 0;
 			for(int i = 0; i < nthreads; i++) {
 				reduceSum_scanned[i] = temp;
 				temp = temp + reduceSum[i];
 			}
 		}

		//scatter back
		#pragma omp for schedule(static) nowait 
		for(int i = 0; i < n; i++) {
			b[i] += reduceSum_scanned[tid];
		}
	}

	gettimeofday(&end, NULL);

 	return diffTime(end, start);
}

//scan function using OpenMP
//pattern : 0 for inclusive, 1 for exclusive
double scan_omp_MIC(int *a, int *b, int n, int pattern) {

	// kmp_set_defaults("KMP_AFFINITY=compact");
    // kmp_set_defaults("KMP_BLOCKTIME=0");

	struct timeval start, end;

    #pragma offload target(mic) \
    in(a:length(n) alloc_if(1) free_if(0)) \
    in(b:length(n) alloc_if(1) free_if(0))
    {}

	gettimeofday(&start, NULL);

    #pragma offload target(mic) \
    nocopy(a:length(n) ) \
 	nocopy(b:length(n) )
 	{
 		int reduceSum[MAX_THREAD_NUM] = {0};
 		int reduceSum_scanned[MAX_THREAD_NUM] = {0};

 		#pragma omp parallel 
 		{
 			int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int localSum = 0;

            //reduce & local prefix sum 
 			#pragma omp for schedule(static) nowait 
			for(int i = 0; i < n; i++) {
				localSum += a[i];
				b[i] = localSum;
 			}	
 			reduceSum[tid] = localSum;
 			
 			#pragma omp barrier

 			//exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
 			#pragma omp single 
 			{
	 			int temp = 0;
	 			for(int i = 0; i < nthreads; i++) {
	 				reduceSum_scanned[i] = temp;
	 				temp = temp + reduceSum[i];
	 			}
	 		}

 			//scatter back
 			#pragma omp for schedule(static) nowait 
 			for(int i = 0; i < n; i++) {
 				b[i] += reduceSum_scanned[tid];
 			}
 		}
 	}

	gettimeofday(&end, NULL);

 	#pragma offload target(mic) \
    out(a:length(n) alloc_if(0) free_if(1)) \
    out(b:length(n) alloc_if(0) free_if(1))
    {}

 	return diffTime(end, start);
}

//scan function using intrinsics on CPU
//pattern : 0 for inclusive, 1 for exclusive
double scan_ass_CPU(int *a, int *b, int n, int pattern) {
	struct timeval start, end;

	int reduceSum[MAX_THREAD_NUM] = {0};
	int reduceSum_scanned[MAX_THREAD_NUM] = {0};
	__m256i zero = _mm256_set1_epi32(0);
	__m256i mask = _mm256_set1_epi32(0xffffffff);		//highest bit should be 1

	gettimeofday(&start, NULL);
	#pragma omp parallel 
	{
		int nthreads = omp_get_num_threads();
	    int tid = omp_get_thread_num();

	    int localSum = 0;
	    int reducedSum;
	    
	    //reduce & local prefix sum 
		#pragma omp for schedule(static) nowait 
		for(int i = 0; i < n; i += 8) {

			__m256i origin = _mm256_maskload_epi32((a + i),mask);
			__m256i shift = origin;
			__m256i shift2 = origin;
	 		__m256i localSumVec = _mm256_set1_epi32(localSum);

	 		//reduction
	 		// reducedSum = _mm256_reduce_add_epi32(origin);	//not available in AVX2
	 		__m256i ori_reduction = origin;		//only for reduction
	 		ori_reduction = _mm256_hadd_epi32(ori_reduction,ori_reduction);
	 		ori_reduction = _mm256_hadd_epi32(ori_reduction,ori_reduction);
	 		reducedSum = ((int*)&ori_reduction)[0] + ((int*)&ori_reduction)[4];
	 		
	 		if (pattern == 1)	//exclusive
	 			origin = _mm256_alignr_epi8(origin, zero, 7);		//this is not tested yet, is wrong!!

	 		//shift 1 lane + add
	 		shift = _mm256_permute2x128_si256(shift, shift, 41);
	 		shift2 = _mm256_alignr_epi8(shift2,shift, 12);
	 		origin = _mm256_add_epi32(origin, shift2);

	 		//shift 2 lanes + add
	 		shift = _mm256_permute2x128_si256(origin, origin, 41);
	 		shift2 = _mm256_alignr_epi8(origin,shift, 8);
	 		origin = _mm256_add_epi32(origin, shift2);

	 		//shift 4 lanes + add
	 		origin = _mm256_add_epi32(_mm256_permute2x128_si256(origin, origin, 41), origin);

	 		//add previous accumulated sum to the current vector
	 		origin = _mm256_add_epi32(origin, localSumVec);
	 		
			_mm256_maskstore_epi32(b+i,mask, origin);

			localSum += reducedSum;
		}	

		reduceSum[tid] = localSum;
		
		#pragma omp barrier

		//exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
		#pragma omp single 
		{
			int temp = 0;
			for(int i = 0; i < nthreads; i++) {
				reduceSum_scanned[i] = temp;
				temp = temp + reduceSum[i];
			}
		}

		//scatter back
		__m256i localSumVec = _mm256_set1_epi32(reduceSum_scanned[tid]);

		#pragma omp for schedule(static) nowait 
		for(int i = 0; i < n; i += 8) {
			__m256i resVec = _mm256_maskload_epi32(b + i, mask);
			resVec = _mm256_add_epi32(resVec,localSumVec);
			_mm256_maskstore_epi32(b+i,mask, resVec);
		}
	}
	gettimeofday(&end, NULL);

 	return diffTime(end, start);
}

//scan function using intrinsics on MIC
//pattern : 0 for inclusive, 1 for exclusive
double scan_ass_MIC(int *a, int *b, int n, int pattern) {
	struct timeval start, end;

    #pragma offload target(mic) \
    in(a:length(n) alloc_if(1) free_if(0)) \
    in(b:length(n) alloc_if(1) free_if(0))
    {}

	gettimeofday(&start, NULL);

    #pragma offload target(mic) \
    nocopy(a:length(n) ) \
 	nocopy(b:length(n) )
 	{
 		int reduceSum[MAX_THREAD_NUM] = {0};
 		int reduceSum_scanned[MAX_THREAD_NUM] = {0};

 		__m512i zero = _mm512_set1_epi32(0);

 		#pragma omp parallel 
 		{
 			int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int localSum = 0;
            int reducedSum;
            
            //reduce & local prefix sum 
 			#pragma omp for schedule(static) nowait 
			for(int i = 0; i < n; i += 16) {

				__m512i origin = _mm512_load_epi32(a + i);
				__m512i shift = origin;
		 		__m512i localSumVec = _mm512_set1_epi32(localSum);

		 		//reduction
		 		reducedSum = _mm512_reduce_add_epi32(origin);	
		 		
		 		if (pattern == 1)	//exclusive
		 			origin = _mm512_alignr_epi32(origin, zero, 15);

		 		shift = _mm512_alignr_epi32(origin,zero, 15);
				origin = _mm512_add_epi32(origin, shift);

		 		shift = _mm512_alignr_epi32(origin,zero, 14);
		 		origin = _mm512_add_epi32(origin, shift);

		 		shift = _mm512_alignr_epi32(origin,zero, 12);
		 		origin = _mm512_add_epi32(origin, shift);

		 		shift = _mm512_alignr_epi32(origin,zero, 8);
		 		origin = _mm512_add_epi32(origin, shift);

		 		//add previous accumulated sum to the current vector
		 		origin = _mm512_add_epi32(origin, localSumVec);

				_mm512_store_epi32((void*)(b+i), origin);

				localSum += reducedSum;
 			}	
 			reduceSum[tid] = localSum;
 			
 			#pragma omp barrier

 			//exclusively scan the reduce sum (at most MAX_THREAD_NUM elements)
 			#pragma omp single 
 			{
	 			int temp = 0;
	 			for(int i = 0; i < nthreads; i++) {
	 				reduceSum_scanned[i] = temp;
	 				temp = temp + reduceSum[i];
	 			}
	 		}

 			//scatter back
 			__m512i localSumVec = _mm512_set1_epi32(reduceSum_scanned[tid]);

 			#pragma omp for schedule(static) nowait 
 			for(int i = 0; i < n; i += 16) {
				__m512i resVec = _mm512_load_epi32(b + i);
				resVec = _mm512_add_epi32(resVec,localSumVec);
				_mm512_store_epi32((void*)(b+i), resVec);
 			}
 		}
 	}

	gettimeofday(&end, NULL);

 	#pragma offload target(mic) \
    nocopy(a:length(n) alloc_if(0) free_if(1)) \
    out(b:length(n) alloc_if(0) free_if(1))
    {}

 	return diffTime(end, start);
}

/* 
 * wrapper functions: function calls & checking functions
 * pattern : 0 for inclusive, 1 for exclusive 
*/

//1. TBB test
double testScan_tbb(int *a, int *b, int n, int pattern) {
	bool res = true;
	int *temp = new int[n];

	double myTime = scan_tbb_CPU(a,b,n, pattern);

	if (pattern == 0) {
		//checking inclusive
		temp[0] = a[0];
		for(int i = 1; i < n; i++)	{
			temp[i] = temp[i-1] + a[i];
			if (b[i] != temp[i])	{
				res = false;
				break;
			}
		}
		// printRes("tbb_scan_inclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	else {
		//checking exclusive
		temp[0] = 0;
		if (temp[0] != b[0])	res = false;
		for(int i = 1; i < n; i++) {
			temp[i] = temp[i-1] + a[i-1];
			if (b[i] != temp[i])	{
				res = false;
				break;
			}
		}
		// printRes("tbb_scan_exclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	delete[] temp;
	return myTime;
}

//2. OpenMP test
double testScan_omp(int *a, int *b, int n, int pattern) {
	bool res = true;
	int *temp = new int[n];

	double myTime = scan_omp_CPU(a,b,n, pattern);

	if (pattern == 0) {
		//checking inclusive
		temp[0] = a[0];
		for(int i = 1; i < 100; i++)	{
			temp[i] = temp[i-1] + a[i];
			if (b[i] != temp[i])	{
				cout<<"different at: "<<i<<endl;
				res = false;
				break;
			}
		}
		// printRes("omp_scan_inclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	else {
		//checking exclusive
		temp[0] = 0;
		if (temp[0] != b[0])	res = false;
		for(int i = 1; i < n; i++) {
			temp[i] = temp[i-1] + a[i-1];
			if (b[i] != temp[i])	{

				res = false;
				break;
			}
		}
		// printRes("omp_scan_exclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	delete[] temp;

	return myTime;
}

//3. Intrinsics test
double testScan_ass(int* a, int* b, int n, int pattern) {
	
	bool res = true;
	int *temp = new int[n];

	double myTime = scan_ass_CPU(a,b,n, pattern);

	if (pattern == 0) {
		//checking inclusive
		temp[0] = a[0];
		for(int i = 1; i < n; i++)	{
			temp[i] = temp[i-1] + a[i];
			if (b[i] != temp[i])	{
				cout<<"different at: "<<i<<endl;
				res = false;
				break;
			}
		}
		// printRes("ass_scan_inclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	else {
		//checking exclusive
		temp[0] = 0;
		if (temp[0] != b[0])	res = false;
		for(int i = 1; i < n; i++) {
			temp[i] = temp[i-1] + a[i-1];
			if (b[i] != temp[i])	{
				res = false;
				break;
			}
		}
		// printRes("ass_scan_exclusive", res, myTime);
		if (!res) {
			cout<<"wrong!"<<endl;
			exit(1);
		}
	}
	delete[] temp;

	return myTime;
}


