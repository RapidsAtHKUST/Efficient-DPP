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

	int *loc = new int[r_len];
	intRandom_Only(loc, r_len,10000);

	double time = 0.0f;

	bool res = testMap(source, r_len, time);
	cout<<"map: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	res = testGather(source, r_len, loc, time);
	cout<<"gather: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	res = testScatter(source, r_len, loc,time);
	cout<<"scatter: ";
	if (res) 	cout<<"Success!"<<'\t';
	else 		cout<<"Fail!"<<'\t';
	cout<<"Time: "<<time<<" ms"<<endl;

	delete[] source;
	delete[] loc;

	return 0;

}