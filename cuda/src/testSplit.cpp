#include "test.h"
using namespace std;

bool testSplit(Record *source, int r_len, double& totalTime,  int fanout, int blockSize, int gridSize) {

	bool res = true;
	
	//allocate for the host memory
	Record *h_source = new Record[r_len];
	Record *h_dest = new Record[r_len];

	for(int i = 0; i < r_len; i++) {
		h_source[i].x = source[i].x;
		h_source[i].y = source[i].y;
		// h_source[i].x = i;
		// h_source[i].y = rand() % 20;
	}
	
	totalTime = splitImpl(h_source, h_dest, r_len, fanout, blockSize, gridSize);

	//checking
    for(int i = 1; i < r_len; i++) {
        if (h_dest[i].y < h_dest[i-1].y)  {
        	res = false;
        	cout<<i<<' '<<h_source[i].x<<' '<<h_source[i].y<<' '<<h_dest[i].x<<' '<<h_dest[i].y<<endl;
        }
    }

    delete[] h_source;
    delete[] h_dest;

    return res;
}