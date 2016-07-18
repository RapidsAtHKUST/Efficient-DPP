#include "utility.h"
#include "functions.h"
#include <immintrin.h>

#pragma offload_attribute(push, target(mic))
#include <iostream>
#include <stdio.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
// #include <dvec.h>

class __attribute__ ((target(mic))) ApplyFoo;

#pragma offload_attribute(pop)

#include "functions.h"

// #include "functions.h"
// #include "tbb/task_scheduler_init.h"
// #include "tbb/blocked_range.h"
// #include "tbb/parallel_for.h"

using namespace std;
// using namespace tbb;


// __attribute__ ((target(mic))) int floorOfPower2_CPU1(int a) {
// 	int base = 1;
// 	while(base < a) {
// 		base <<=1;
// 	}
// 	return base >> 1;
// }

// class __attribute__ ((target(mic))) ApplyFoo
// {
// 	int * my_res;
// 	int * const my_a;
// public: 
// 	void operator() (const blocked_range<size_t> &r) const
// 	{
// 		int *a = my_a;
// 		size_t ended = r.end();
// 		for(size_t i = r.begin(); i < ended; i++ )	
// 			my_res[i] = floorOfPower2_CPU1(a[i]);
// 	}
	
// 	__attribute__ ((target(mic))) ApplyFoo(int a[], int res[]): my_a(a), my_res(res) {}
// };

int main() {

	double totalTime = 0.0f;
	int n = 16000000;

	int *source_intr = (int*)_mm_malloc(sizeof(int)*n, 64);
	int *dest_intr = (int*)_mm_malloc(sizeof(int)*n, 64);
	int *loc_intr = (int*)_mm_malloc(sizeof(int)*n, 64);

	int * source = new int[n];
	int * dest = new int[n];
	int *loc = new int[n];

	for(int i = 0; i < n; i++) {
		source[i] = i;
		source_intr[i] = i;
		loc[i] = n - i - 1;
		loc_intr[i] = n - i - 1;
	}

	// int *a = new int[n];
	// int *b = new int[n];

	for(int i = 0; i < n; i++) {
		// a[i] = rand();
		// b[i] = rand() % 100 + 1;
	}

	// cout<<endl;
	// task_scheduler_init init;
	// struct timeval start, end;
	// int *a = new int[n];
	// int *res = new int[n];

	// for(int i = 0; i < n; i++) {
	// 	a[i] = i;
	// }
	// 	printf("starting for,\n");

	// #pragma offload target(mic) in(a:length(n)) out(res:length(n))
	// {
	// 	gettimeofday(&start, NULL);
	// 	printf("start for,\n");
	// 	parallel_for(blocked_range<size_t>(0,n), ApplyFoo(a, res), simple_partitioner());
	// 	printf("end for,\n");

	// 	gettimeofday(&end, NULL);
	// }

	// cout<<"Output:"<<endl;

	// for(int i = 0; i < n ; i++) {
	// 	cout<<a[i]<<' '<<res[i]<<endl;
	// }

	// return diffTime(end, start);

	// cout<<"Elapsed time for mapping "<<n<<" elements: "<<diffTime(end, start)<<" ms."<<endl;


	// cout<<"1------------------------------------------"<<endl;
	// totalTime = testScan(a, b, n);
	// cout<<"Elapsed time for mapping "<<n<<" elements: "<<totalTime<<" ms."<<endl;
	// cout<<"2------------------------------------------"<<endl;
	// testMap(a,b,n);
	// cout<<"3------------------------------------------"<<endl;
	// totalTime = testScan(a,b,n);
	// cout<<"Elapsed time for mapping "<<n<<" elements: "<<totalTime<<" ms."<<endl;
	// cout<<"4------------------------------------------"<<endl;
	// testMap(a,b,n);

	// cout<<"input: ";
	// 	for(int j = 0; j < n; j++) {
	// 		a[j] = j + j ;
	// 		cout<<a[j]<< ' ';
	// 	}
	// cout<<endl;

	int num = 10;
	double lessTime = 9999;

	for(int i = 0; i < num; i++) {
		// for(int j = 0; j < n; j++) {
		// 	a[j] = rand() % 10000;
		// }
		double myTime = map(source_intr,dest_intr,n);

		bool res = true;
		for(int i = 0; i < n; i++) {
			if (dest_intr[i] != floorOfPower2_CPU(source_intr[i]))	{
				res = false;
				break;
			}
		}
		if (res)	cout<<"True"<<endl;
		else 		cout<<"False"<<endl;
	// 	cout<<"gather:"<<endl;
		// testGather_intr(source, dest, loc, n);
	// 	cout<<endl;

	// 	cout<<"gather_intr:"<<endl;
	//	testGather_intr(source_intr, dest_intr, loc_intr, n);
	// 	cout<<endl;

	// 	cout<<"scatter:"<<endl;
	// 	testScatter(source, dest, loc, n);
	// 	cout<<endl;

	// 	cout<<"scatter_intr:"<<endl;
	// 	testScatter_intr(source_intr, dest_intr, loc_intr, n);
	// 	cout<<endl;
		// double myTime = testScan_omp(a,b,n,0);
		// testScan_tbb(a,b,n,0);
		// double myTime = testScan_ass(source_intr,dest_intr,n,0);
		if (myTime < lessTime) {
			lessTime = myTime;
		}
		 // testRadixSort(a, n);
		 // testRadixSort_tbb(a, n);

		// cout<<"output: ";
		// for(int j = 0; j <n ; j++) {
		// 	cout<<b[j]<<' ';
		// }
		// cout<<endl;	
		// cout<<"output: ";
		// for(int j = 0; j < n; j++) {
		// 	cout<<a[j]<<' ';
		// }
		// cout<<endl;
	}
	
	cout<<"Time:"<<lessTime<<" ms"<<endl;
	// delete[] a;
	// delete[] b;
	// cout<<"Avg time: "<<totalTime /  (num - 1) <<" ms."<<endl;
	_mm_free(source_intr);
	_mm_free(dest_intr);
	_mm_free(loc_intr);

	delete[] source;
	delete[] dest;
	delete[] loc;
	
	return 0;
}
