#pragma offload_attribute(push, target(mic))
#include <iostream>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/parallel_for.h"
#pragma offload_attribute(pop)

#include "functions.h"

using namespace std;
using namespace tbb;


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
 		// y[0] = 0;
 		for( int i=r.begin(); i<r.end(); ++i ) {
 			if( Tag::is_final_scan() )
 				y[i] = temp;
 			// if (i == 0)	continue;
 			temp = temp + x[i];
 			
 		}
 		sum = temp;
 	}
 	ScanBody_ex( ScanBody_ex& b, split ) : x(b.x), y(b.y), sum(0) {}
 	void reverse_join( ScanBody_ex& a ) { sum = a.sum + sum;}
 	void assign( ScanBody_ex& b ) {sum = b.sum;}
};

double scan_tbb(int *a, int *b, int n) {
	task_scheduler_init init;
	struct timeval start, end;

	#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
	{
		gettimeofday(&start, NULL);
		ScanBody_ex<int> body(b,a);
 		parallel_scan( blocked_range<int>(0,n), body );
		gettimeofday(&end, NULL);
	}
	return diffTime(end, start);
}

void testScan_tbb(int *a, int *b, int n) {
	bool res = true;
	double myTime = scan_tbb(a,b,n);
	int *temp = new int[n];

	//checking inclusive
	temp[0] = a[0];
	for(int i = 1; i < n; i++)	{
		temp[i] = temp[i-1] + a[i];
		if (b[i] != temp[i])	{
			res = false;
			break;
		}
	}

	//checking exclusive
	cout<<"temp:";
	temp[0] = 0;
	cout<<temp[0]<<' ';
	for(int i = 1; i < n; i++) {
		temp[i] = temp[i-1] + a[i-1];
		cout<<temp[i]<<' ';
	}
	cout<<endl;
	printRes("tbb_scan", res, myTime);

	delete[] temp;
}