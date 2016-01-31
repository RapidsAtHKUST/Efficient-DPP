#include "test.h"
using namespace std;

bool testBisort(Record *source, int r_len, double& totalTime,int dir, int blockSize, int gridSize) {
	bool res = true;

	Record *h_source = new Record[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source[i] = source[i];
	}

	totalTime = bitonicSortImpl(h_source, r_len, dir, blockSize, gridSize);

	//testing
	// cout<<"Output:"<<endl;
	// for(int i = 0; i < r_len; i++) {
	// 	cout<<h_source[i].x<<' '<<h_source[i].y<<endl;
	// }

	for(int i = 0; i < r_len-1; i++) {
		if( (h_source[i].y > h_source[i+1].y && dir == 1) ||
			(h_source[i].y < h_source[i+1].y && dir == 0) )	
			res = false;
	}

	return res;

}