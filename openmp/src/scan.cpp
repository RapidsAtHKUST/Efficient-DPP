#pragma offload_attribute(push, target(mic))
#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/parallel_for.h"
#include <immintrin.h>
#pragma offload_attribute(pop)

#include "functions.h"

using namespace std;
using namespace tbb;

#define MAX_THREAD_NUM	(256)

//tbb scan (inclusive) reach 80% utilization
template<typename T> class __attribute__ ((target(mic))) ScanBody_in {
 	T sum;
 	T* const y;
 	const T* const x; 
public:
 	ScanBody_in( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
 	T get_sum() const {return sum;}
 
 	template<typename Tag>
 	void operator()( const blocked_range<int>& r, Tag ) {
 		T temp = sum;
 		for( int i=r.begin(); i<r.end(); ++i ) {
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

template<typename T> class __attribute__ ((target(mic))) ScanBody_ex {
 	T sum;
 	T* const y;
 	const T* const x; 
public:
 	ScanBody_ex( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
 	T get_sum() const {return sum;}
 
 	template<typename Tag>
 	void operator()( const blocked_range<int>& r, Tag ) {
 		T temp = sum;
 		for( int i=r.begin(); i<r.end(); ++i ) {
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

double scan_tbb(int *a, int *b, int n, int pattern) {
	task_scheduler_init init;
	struct timeval start, end;

	if (pattern == 1) {
		#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
		{
			gettimeofday(&start, NULL);
			ScanBody_ex<int> body(b,a);
	 		parallel_scan( blocked_range<int>(0,n), body );
			gettimeofday(&end, NULL);
		}
	}
	else {
		#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
		{
			gettimeofday(&start, NULL);
			ScanBody_in<int> body(b,a);
	 		parallel_scan( blocked_range<int>(0,n), body );
			gettimeofday(&end, NULL);
		}
	}
	return diffTime(end, start);
}

//pattern : 0 for inclusive, 1 for exclusive
void testScan_tbb(int *a, int *b, int n, int pattern) {
	bool res = true;
	int *temp = new int[n];

	double myTime = scan_tbb(a,b,n, pattern);

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
		printRes("tbb_scan_inclusive", res, myTime);
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
		printRes("tbb_scan_exclusive", res, myTime);
	}
	delete[] temp;
}

double scan_omp(int *a, int *b, int n, int pattern) {

	// kmp_set_defaults("KMP_AFFINITY=compact");
 //    kmp_set_defaults("KMP_BLOCKTIME=0");

	struct timeval start, end;

    #pragma offload target(mic) \
    in(a:length(n) alloc_if(1) free_if(1)) \
 	out(b:length(n) alloc_if(1) free_if(1))
 	{
		gettimeofday(&start, NULL);

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

 			// __m512i offset = _mm512_set1_epi32(reduceSum_scanned[tid]);
    //     	#pragma omp for schedule(static) //second parallel pass with SSE as well
    //     	for (int i = 0; i<n; i+=16) {       
	   //          __m512i tmp1 = _mm512_load_epi32( b + i );
	   //          tmp1 = _mm512_add_epi32(tmp1, offset);    
	   //          _mm512_store_epi32(b + i, tmp1);
    //     	}
 		}
		gettimeofday(&end, NULL);
 	}
 	return diffTime(end, start);
}

//pattern : 0 for inclusive, 1 for exclusive
void testScan_omp(int *a, int *b, int n, int pattern) {
	bool res = true;
	int *temp = new int[n];

	double myTime = scan_omp(a,b,n, pattern);

	if (pattern == 0) {
		//checking inclusive
		temp[0] = a[0];
		for(int i = 1; i < 100; i++)	{
			temp[i] = temp[i-1] + a[i];
			// cout<<b[i]<<' '<<temp[i]<<endl;
			if (b[i] != temp[i])	{
				cout<<"different at: "<<i<<endl;
				res = false;
				break;
			}
		}
		printRes("omp_scan_inclusive", res, myTime);

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
		printRes("omp_scan_exclusive", res, myTime);
	}
	delete[] temp;
}


