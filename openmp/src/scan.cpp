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

template<typename T> class __attribute__ ((target(mic))) ScanBody {
 	T sum;
 	T* const y;
 	const T* const x; 
public:
 	ScanBody( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
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
 	ScanBody( ScanBody& b, split ) : x(b.x), y(b.y), sum(0) {}
 	void reverse_join( ScanBody& a ) { sum = a.sum + sum;}
 	void assign( ScanBody& b ) {sum = b.sum;}
};

double scan_tbb(int *a, int *b, int n) {
	task_scheduler_init init;
	struct timeval start, end;

	#pragma offload target(mic) in(a:length(n) alloc_if(1) free_if(0)) out(b:length(n) alloc_if(1) free_if(0))
	{
		gettimeofday(&start, NULL);
		ScanBody<int> body(b,a);
 		parallel_scan( blocked_range<int>(0,n,1000), body );
		gettimeofday(&end, NULL);
	}
	return diffTime(end, start);
}









void testScan_tbb(int *a, int *b, int n) {
	double myTime = scan_tbb(a,b,n);
	int *temp = new int[n];

	//checking
	bool res = true;
	temp[0] = a[0];
	for(int i = 1; i < n; i++)	{
		temp[i] = temp[i-1] + a[i];
		if (b[i] != temp[i])	{
			res = false;
			break;
		}
	}
	cout<<"Num: "<<n<<endl;
	cout<<"Time: "<<myTime<<" ms."<<endl;
	if (res)	cout<<"Pass!"<<endl;
	else		cout<<"Failed!"<<endl;

	delete[] temp;
}