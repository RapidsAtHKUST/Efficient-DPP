#include "functions.h"
using namespace std;


//template function pointer
// template<class T>
// struct Type {
// 	typedef T (*Function)(T);
// };

//mapping function
__attribute__ ((target(mic))) int floorOfPower2(int a) {
	int k = 1;
	while (k < a) k <<=1;
	return k>>1;
}

// template<typename T, typename FuncType>
double map(int *source, int *dest, int n /*, FuncType func*/) {
	// #pragma offload target(mic) \
	// in(source:length(n) alloc_if(1) free_if(0)) \
	// nocopy(dest:length(n) alloc_if(1) free_if(0))
	// {};
	
    kmp_set_defaults("KMP_AFFINITY=compact");

    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1))  
    {
    	gettimeofday(&start, NULL);
		#pragma omp parallel for
		for(int i = 0; i < n; i++) {
			dest[i] = floorOfPower2(source[i]);
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	return diffTime(end,start);
}

void testMap() {

	// Type<int>::Function map1 = floorOfPower2;
	int n = 16000000;
	int *a = new int[n];
	int *b = new int[n];

	for(int i = 0; i < n; i++) {
		a[i] = rand() % (INT_MAX/2);
	}

	// double myTime = map<int,Type<int>::Function>(a, b, n, map1);
	double myTime = map(a,b,n);

	//checking
	bool res = true;
	for(int i = 0; i < n; i++) {
		if (b[i] != floorOfPower2(a[i]))	{
			res = false;
			break;
		}
	}

	cout<<"Num: "<<n<<endl;
	cout<<"Time: "<<myTime<<" ms."<<endl;
	if (res)	cout<<"Right!"<<endl;
	else		cout<<"Wrong!"<<endl;

	delete[] a;
	delete[] b;
}
