//
//  utility.cpp
//  gpuqp_cuda
//
//  Created by Zhuohang Lai on 01/19/2016.
//  Copyright (c) 2015-2016 Zhuohang Lai. All rights reserved.
//
#include "utility.h"
using namespace std;

double diffTime(struct timeval end, struct timeval start) {
	return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

void my_itoa(int num, char *buffer, int base) {
    int len=0, p=num;
    while(p/=base) len++;
    len++;
    for(p=0;p<len;p++)
    {
        int x=1;
        for(int t=p+1;t<len;t++) x*=base;
        buffer[p] = num/x +'0';
        num -=(buffer[p]-'0')*x;
    }
    buffer[len] = '\0';
}

//calculating the memory bandwidth
//elasped time: in ms using diffTime
double compute_bandwidth(unsigned long num, int wordSize, double kernel_time) {
    return (double)(1.0*num)/1024/1024/1024*wordSize/kernel_time * 1000 ;
}

bool pair_cmp (pair<double, double> i , pair<double, double> j) {
    return i.first < j.first;
}

double average_Hampel(double *input, int num) {
    int valid = 0;
    double total = 0;

    double *temp_input = new double[num];
    vector< pair<double, double> > myabs_and_input_list;

    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++)
        myabs_and_input_list.push_back(make_pair(fabs(temp_input[i]-mean),temp_input[i]));

    typedef vector< pair<double, double> >::iterator VectorIterator;
    sort(myabs_and_input_list.begin(), myabs_and_input_list.end(), pair_cmp);

    if (num % 2 == 0)  abs_mean = 0.5*(myabs_and_input_list[num/2-1].first + myabs_and_input_list[num/2].first);
    else               abs_mean = myabs_and_input_list[(num-1)/2].first;

    abs_mean /= 0.6745;

    for(VectorIterator iter = myabs_and_input_list.begin(); iter != myabs_and_input_list.end(); iter++) {
        if (abs_mean == 0) { /*if abs_mean=0,only choose those with abs=0*/
            if (iter->first == 0) {
                total += iter->second;
                valid ++;
            }
        }
        else {
            double div = iter->first / abs_mean;
            if (div <= 3.5) {
                total += iter->second;
                valid ++;
            }
        }
    }
    total = 1.0 * total / valid;
    if(temp_input)  delete[] temp_input;
    return total;
}