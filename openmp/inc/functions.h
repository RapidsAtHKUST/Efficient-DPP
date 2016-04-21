#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "utility.h"



void testRadixSort(unsigned *arr, int len) ;
void testRadixSort_tbb(unsigned *arr_tbb, int len) ;


void testGather(int *source, int *dest, int *loc, int n);
void testGather_intr(int *source, int *dest, int *loc, int n);

void testScatter(int *source, int *dest, int* loc, int n);
void testScatter_intr(int *source, int *dest, int* loc, int n);

void testScan_tbb(int* a, int* b, int n, int pattern);
void testScan_omp(int *a, int *b, int n, int pattern);

double map(float *source, float *dest, int n);

#endif