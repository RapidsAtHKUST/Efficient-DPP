#include "test.h"

bool testMap(Record *source, int r_len, int blockSize, int gridSize) {
	
	bool res = true;

	//allocate for the host memory
	Record *h_source = new Record[r_len];
	Record *h_res = new Record[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source[i].x = source[i].x;
		h_source[i].y = source[i].y;
	}

	mapImpl(h_source, h_res, r_len, blockSize, gridSize);

	//checking 
	for(int i = 0; i < r_len; i++) {
		if (h_source[i].x != h_res[i].x ||
			floorOfPower2_CPU(h_source[i].y) != h_res[i].y) 
				res = false;
	}

	delete[] h_source;
	delete[] h_res;

	return res;
}