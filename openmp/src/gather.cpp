#include "functions.h"
using namespace std;

double gather(Record *source, Record *dest, int n, int *loc /*, FuncType func*/) {
	// #pragma offload target(mic) \
	// in(source:length(n) alloc_if(1) free_if(0)) \
	// nocopy(dest:length(n) alloc_if(1) free_if(0))
	// {};
	
    kmp_set_defaults("KMP_AFFINITY=compact");

    struct timeval start, end;

    #pragma offload target(mic) \
    in(source:length(n) alloc_if(1) free_if(1)) \
    out(dest:length(n) alloc_if(1) free_if(1)) \
    in(loc:length(n) alloc_if(1) free_if(1))
    {
    	gettimeofday(&start, NULL);
		#pragma omp parallel for
		for(int i = 0; i < n; i++) {
			dest[i] = source[loc[i]];
		}
		#pragma omp barrier
    	gettimeofday(&end, NULL);
	}

	return diffTime(end,start);
}

void testGather() {
	int n = 16000000;
	Record *source = new Record[n];
	Record *dest = new Record[n];

	int *loc = new int[n];
	intRandom_Only(loc, n, n/2);

	recordRandom(source, n, 10000000);

	double myTime = gather(source, dest, n, loc);

	//checking
	bool res = true;
	for(int i=  0; i < n; i++) {
		if (dest[i].x != source[loc[i]].x ||
			dest[i].y != source[loc[i]].y)	{
			res = false;
			break;
		}
	}
	cout<<"Num: "<<n<<endl;
	cout<<"Time: "<<myTime<<" ms."<<endl;
	if (res)	cout<<"Right!"<<endl;
	else		cout<<"Wrong!"<<endl;

	delete[] source;
	delete[] dest;
	delete[] loc;
}