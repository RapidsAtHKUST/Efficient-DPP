//
//  utility.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/2016.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "utility.h"

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
double computeMem(unsigned long num, int wordSize, double kernel_time) {
    return (double)(1.0*num)/1024/1024/1024*wordSize/kernel_time * 1000 ;
}

double averageHampel(double *input, int num) {
    int valid = 0;
    double total = 0;

    if (num == 1)   return input[0];

    double *temp_input = new double[num];
    double *myabs = new double[num];
    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    std::sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++)    myabs[i] = fabs(temp_input[i]-mean);

    std::sort(myabs, myabs+num);
    if (num % 2 == 0)  abs_mean = 0.5*(myabs[num/2-1] + myabs[num/2]);
    else               abs_mean = myabs[(num-1)/2];

    abs_mean /= 0.6745;

    for(int i = 0; i < num; i++) {
        double div = myabs[i] / abs_mean;
        if (div <= 3.5) {
            total += temp_input[i];
            valid ++;
        }
    }
    total = 1.0 * total / valid;

    if(temp_input)  delete[] temp_input;
    if (myabs)      delete[] myabs;
    return total;
}