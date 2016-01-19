#include "utility.h"
#include "kernels.h"
using namespace std;

int main(int argc, char *argv[]) {
	
	Record *h_source, *h_res;

	int r_len = atoi(argv[1]);
	int max = 10000000;

	//allocate for the host memory
	h_source = (Record*)malloc(sizeof(Record) * r_len);
	h_res = (Record*)malloc(sizeof(Record) * r_len);

	srand(NULL);
	for(int i = 0; i < r_len; i++) {
		h_source[i].x = i;
		h_source[i].y = rand() % max;
	}

	mapImpl(h_source, h_res, r_len);

	for(int i = 0; i < r_len; i++) {
		cout<<h_source[i].x<<' '<<h_source[i].y<<' '<<h_res[i].x<<' '<<h_res[i].y<<endl;
	}

	free(h_source);
	free(h_res);

	return 0;

}