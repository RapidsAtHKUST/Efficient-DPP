#include "test.h"

using namespace std;

bool testScan(int *source, int r_len, double& totalTime, int isExclusive,  int blockSize, int gridSize) {
	
	bool res = true;
	
	//allocate for the host memory
	int *h_source_gpu = new int[r_len];
	int *h_source_cpu = new int[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source_gpu[i] = source[i];
		h_source_cpu[i] = source[i];
	}
	std::cout<<std::endl;
	
	totalTime = scanImpl(h_source_gpu, r_len, blockSize, gridSize, isExclusive);

	// checking 

	if (isExclusive == 0) {         //inclusive
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = source[i] + h_source_cpu[i-1];
        }
    }
    else {                          //exclusive
        h_source_cpu[0] = 0;
        for(int i = 1 ; i < r_len; i++) {
            h_source_cpu[i] = h_source_cpu[i-1] + source[i-1];
        }
    }
    
    for(int i = 0; i < r_len; i++) {
        if (h_source_cpu[i] != h_source_gpu[i]) res = false;
    }

	if (res)	cout<<"Pass!"<<endl;
	else		cout<<"Failed!"<<endl;

	delete[] h_source_gpu;
	delete[] h_source_cpu;
	
	return res;
}