#include "test.h"

bool testScatter(Record *source, int r_len, int *loc,double& totalTime, int blockSize, int gridSize) {
	
	bool res = true;

	//allocate for the host memory
	Record *h_source = new Record[r_len];
	Record *h_res = new Record[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source[i].x = source[i].x;
		h_source[i].y = source[i].y;
	}

	totalTime = scatterImpl(h_source, h_res, r_len, loc,  blockSize, gridSize);

	//checking 
	for(int i = 0; i < r_len; i++) {
		if (h_res[loc[i]].x != h_source[i].x ||
			h_res[loc[i]].y != h_source[i].y) {
			res = false;
			break;
		}
	}

	delete[] h_source;
	delete[] h_res;

	return res;
}