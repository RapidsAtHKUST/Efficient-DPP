#include "utility.h"
#include <stdlib.h>
#include "functions.h"
#include <immintrin.h>
#include <math.h>


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

	double readTime = 9999.0;
	double writeTime = 9999.0;
	double mulTime = 9999.0;
	double addTime = 9999.0;
	double totalTime;

	// int n = atoi(argv[1]);

	// cout<<"num:"<<n<<endl;

cout<<"---------------- begin test ----------------"<<endl;

	// for(int i = 0; i < expr; i++) {
	// 	totalTime = mem_read_test(input, n);
	// 	if (totalTime < readTime) readTime = totalTime;
	// }
	// cout<<"read test: finished."<<endl;

	// for(int i = 0; i < expr; i++) {
	// 	totalTime = mem_write_test(output, n);
	// 	if (totalTime < writeTime) writeTime = totalTime;
	// }
	// cout<<"write test: finished."<<endl;

	// for(int i = 0; i < expr; i++) {
	// 	totalTime = mem_mul_test(input, output, n);
	// 	if (totalTime < mulTime) mulTime = totalTime;
	// }

	// for(int i = 0; i < expr; i++) {
	// 	totalTime = mem_add_test(input,input_2, output, n);
	// 	if (totalTime < addTime) addTime = totalTime;
	// }

	// cout<<"mul test: finished"<<endl;

	// cout<<"Read totalTime:"<<readTime<<" ms.\t"
	// 	<<"Read Throughput: "<< computeMem(n, sizeof(int), readTime)<<" GB/s"<<endl;
	// cout<<"Write totalTime:"<<writeTime<<" ms.\t"
	// <<"Write Throughput: "<< computeMem(n, sizeof(int), writeTime)<<" GB/s"<<endl;
	// cout<<"Mul totalTime:"<<mulTime<<" ms.\t"
	// <<"Mul Throughput: "<< computeMem(n*2, sizeof(int), mulTime)<<" GB/s"<<endl;
	// cout<<"Add totalTime:"<<addTime<<" ms.\t"
	// <<"Add Throughput: "<< computeMem(n*3, sizeof(int), addTime)<<" GB/s"<<endl;

	// delete[] input;
	// delete[] input_2;
	// delete[] output;
	// _mm_free(input);
	// _mm_free(output);
//----------------------------------- scan testing -----------------------------------
	int expr = 10;

	//initialization for SSE & AVX instructions

	//initialiization for normal instructions
	int dataSize[11] = {1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000,512000000,1000000000}; 

	for(int d = 0; d < 11; d++) {
		double *allTime = new double[expr];			//for storing the times

		int cur_size = dataSize[d];
		int *source = new int[cur_size];
		// int* source = (int*)_mm_malloc(sizeof(int)*cur_size, 64);			//for AVX2

		for(int i = 0; i < cur_size; i++)	{
			source[i] = 1;
		}

		for(int j = 0; j < expr; j++) {
			int *dest = new int[cur_size];
			// int* dest = (int*)_mm_malloc(sizeof(int)*cur_size, 64);

			allTime[j] = testScan_omp(source,dest,cur_size,0);	

			// _mm_free(dest);
			delete[] dest;
		}
		delete[] source;
		// _mm_free(source);
		sort(allTime,allTime+expr);

		cout<<"size: "<<cur_size<<'\t'<<"time: "<<(allTime[expr/2-1]+allTime[expr/2])/2<<" ms"<<endl;
		delete[] allTime;
	}
	

	
	
	//test function
	
	//free for SSE & AVX instructions
	// _mm_free(source);
	// _mm_free(dest);

	//free for normal instructions

//----------------------------------- snd of scan testing -----------------------------------
	

// int typeOfDevice = atoi(argv[2]);

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
// if (typeOfDevice == 0)	//cpu
// 	map_CPU(source,dest,n);
// else if (typeOfDevice == 1)
// 	map_MIC(source,dest,n,0);
// else {
// 	cerr<<"Invalid type!"<<endl;
// 	return 1;	
// }

	// int numOfTests = 3;

	// bytes[0] = n * sizeof(float) * 2;
	// for(int k = 0; k < NUM_FUCTIONS; k++) {
	// 	for(int i = 0; i < numOfTests; i++) {

	// 		double tempTime;		
	// 		if (typeOfDevice == 0)	//cpu
	// 			tempTime = map_CPU(source,dest,n);
	// 		else if (typeOfDevice == 1)
	// 			tempTime = map_MIC(source,dest,n,k*STEP);

	// 		cout<<"current: k="<<k*STEP<<' '<<"time: "<<tempTime<<" ms."<<endl;
	// 		if (tempTime < minTime[k])	minTime[k] = tempTime;
	// 		if (tempTime > maxTime[k])	maxTime[k] = tempTime;
	// 		avgTime[k] += tempTime;
	// 	}
	// 	avgTime[k] /= numOfTests;
	// }

	//checking
	// bool res = true;
	// for(int i = 0; i < n; i++) {
	// 	if (dest[i] != source[i] + 1) {
	// 		res = false;
	// 		break;
	// 	}
	// }
	// cout<<endl;

	// cout<<"------------------------------------"<<endl;
	// cout<<"Summary:"<<endl;
	// if (typeOfDevice == 0)
	// 	cout<<"Platrform: CPU"<<endl;
	// else if (typeOfDevice == 1)
	// 	cout<<"Platrform: MIC"<<endl;
	// if (res)	cout<<"Output:right"<<endl;
	// else 		cout<<"Output:wrong"<<endl;

	// cout<<"Number of tuples: "<<n<<endl;

	// for(int k = 0; k < NUM_FUCTIONS; k++) {
	// 	cout<<"k = "<<k*STEP<<' '
	// 		<<"Avg Time: "<<avgTime[k]<<" ms."<<' '
	// 		<<"Min Time: "<<minTime[k]<<" ms."<<' '
	// 		<<"Max Time: "<<maxTime[k]<<" ms."<<' '
	// 		<<"Rate: "<<bytes[0] / minTime[k] * 1.0E-06<<" GB/s."<<endl;
	// }

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
	
	return 0;

	
}
