#include "test.h"
using namespace std;

bool testRadixSort(Record *source, int r_len, double& totalTime, int blockSize, int gridSize) {
	bool res = true;
	Record *h_source = new Record[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source[i].x = source[i].x;
		h_source[i].y = source[i].y;
	}

	totalTime = radixSortImpl(h_source, r_len, blockSize, gridSize);

	cout<<"Output:"<<endl;
	for(int i = 0; i < r_len-1; i++) {
		if (h_source[i].y > h_source[i+1].y) res = false;
	}
	return res;
}