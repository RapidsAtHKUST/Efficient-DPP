//
//  main.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "test.h"

using namespace std;

int main(int argc, char *argv[]) {
	
	int r_len = atoi(argv[1]);
	Record *source = new Record[r_len];
	recordRandom(source, r_len);

	bool res = testMap(source, r_len);

	if (res) 	cout<<"Success!"<<endl;
	else 		cout<<"Fail!"<<endl;

	delete[] source;
	return 0;

}