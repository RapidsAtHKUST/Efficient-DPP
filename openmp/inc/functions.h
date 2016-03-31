#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "utility.h"



void testRadixSort();

void testGather(int *source, int *dest, int *loc, int n);
void testGather_intr(int *source, int *dest, int *loc, int n);

double testScatter(int *source, int *dest, int* loc, int n);
double testScatter_intr(int *source, int *dest, int* loc, int n);

void testScan_tbb(int*, int*, int);

double map(float *source, float *dest, int n);

#endif