#include "utility.h"
#include <stdlib.h>
#include "functions.h"
#include <immintrin.h>

// #pragma offload_attribute(push, target(mic))
// #include <iostream>
// #include <stdio.h>
// #include "tbb/task_scheduler_init.h"
// #include "tbb/blocked_range.h"
// #include "tbb/parallel_for.h"


// #pragma offload_attribute(pop)
#define NUM_FUCTIONS	(25)		//map, scatter, gather, reduce, scan, split
#define STEP			(10)
#define MAX_TIME_INIT 		(99999.0)
#define MIN_TIME_INIT		(0.0)

using namespace std;

double bytes[NUM_FUCTIONS];
double minTime[NUM_FUCTIONS]={MAX_TIME_INIT};
double maxTime[NUM_FUCTIONS]={MIN_TIME_INIT};
double avgTime[NUM_FUCTIONS]={0};


int main(int argc, char* argv[]) {

	double totalTime = 0.0f;
	int n = atoi(argv[1]);

	cout<<"num:"<<n<<endl;

	float *input = (float*)_mm_malloc(sizeof(float)*n, 64);
	for(int i = 0; i < n;i++) {
		input[i] = 0;
	}


cout<<"---------------- begin test ----------------"<<endl;
	
	double lessTime = 9999999;

	int expr = 2;
	int repeatTime = 1;

	for(int i = 0; i < expr; i++) {
		totalTime = mad_test(input,n,1);
		if (totalTime < lessTime) lessTime = totalTime;
	}
	cout<<"totalTime:"<<lessTime<<" ms."<<endl;
	cout<<"Throughput: "<< (double)n * 2 * repeatTime * 240 / lessTime * 0.001 * 0.001<<" GFLPS"<<endl;

	_mm_free(input);

	return 0;

/*
	int *source, *dest;

	// source = (int*)_mm_malloc(sizeof(int)*n, 64);
	// dest = (int*)_mm_malloc(sizeof(int)*n, 64);

	source = new int[n];
	dest = new int[n];

	//initialization
	for(int i = 0; i < n; i++) {
		source[i] = 1;
		dest[i] = -1;
	}

int typeOfDevice = atoi(argv[2]);

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

	//warm up
if (typeOfDevice == 0)	//cpu
	map_CPU(source,dest,n);
else if (typeOfDevice == 1)
	map_MIC(source,dest,n,0);
else {
	cerr<<"Invalid type!"<<endl;
	return 1;	
}

	int numOfTests = 3;

	bytes[0] = n * sizeof(float) * 2;
	for(int k = 0; k < NUM_FUCTIONS; k++) {
		for(int i = 0; i < numOfTests; i++) {

			double tempTime;		
			if (typeOfDevice == 0)	//cpu
				tempTime = map_CPU(source,dest,n);
			else if (typeOfDevice == 1)
				tempTime = map_MIC(source,dest,n,k*STEP);

			cout<<"current: k="<<k*STEP<<' '<<"time: "<<tempTime<<" ms."<<endl;
			if (tempTime < minTime[k])	minTime[k] = tempTime;
			if (tempTime > maxTime[k])	maxTime[k] = tempTime;
			avgTime[k] += tempTime;
		}
		avgTime[k] /= numOfTests;
	}

	//checking
	bool res = true;
	// for(int i = 0; i < n; i++) {
	// 	if (dest[i] != source[i] + 1) {
	// 		res = false;
	// 		break;
	// 	}
	// }
	// cout<<endl;

	cout<<"------------------------------------"<<endl;
	cout<<"Summary:"<<endl;
	if (typeOfDevice == 0)
		cout<<"Platrform: CPU"<<endl;
	else if (typeOfDevice == 1)
		cout<<"Platrform: MIC"<<endl;
	if (res)	cout<<"Output:right"<<endl;
	else 		cout<<"Output:wrong"<<endl;

	cout<<"Number of tuples: "<<n<<endl;

	for(int k = 0; k < NUM_FUCTIONS; k++) {
		cout<<"k = "<<k*STEP<<' '
			<<"Avg Time: "<<avgTime[k]<<" ms."<<' '
			<<"Min Time: "<<minTime[k]<<" ms."<<' '
			<<"Max Time: "<<maxTime[k]<<" ms."<<' '
			<<"Rate: "<<bytes[0] / minTime[k] * 1.0E-06<<" GB/s."<<endl;
	}

		// bool res = true;
		// for(int i = 0; i < n; i++) {
		// 	if (dest[i] != floorOfPower2_CPU(source[i]) )	{
		// 		res = false;
		// 		break;
		// 	}
		// }
		// if (res)	cout<<"True"<<endl;
		// else 		cout<<"False"<<endl;
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
		// for(int i=0; i < 240; i++) {
		// 	cout<<i<<':'<<dest[i]<<endl;
		// }

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
	
	// delete[] a;
	// delete[] b;
	// cout<<"Avg time: "<<totalTime /  (num - 1) <<" ms."<<endl;
	
	// _mm_free(loc_intr);

	// _mm_free(source);
	// _mm_free(dest);
	delete[] source;
	delete[] dest;
	// delete[] loc;
	
	return 0;

	*/
}
