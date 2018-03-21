#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "utility.h"


//basic operations
double vpu_test(float *hostMem, int n, int repeatTime);
double mem_read_test(int *data, int n);
double mem_write_test(int *data, int n);
double mem_mul_test(int *input, int*output, int n);
double mem_add_test(int *input,int *input_2,  int*output, int n);

void testRadixSort(unsigned *arr, int len) ;
void testRadixSort_tbb(unsigned *arr_tbb, int len) ;

void testGather(int *source, int *dest, int *loc, int n);
void testGather_intr(int *source, int *dest, int *loc, int n);

void testScatter(int *source, int *dest, int* loc, int n);
void testScatter_intr(int *source, int *dest, int* loc, int n);

double testScan_tbb(int* a, int* b, int n, int pattern);
double testScan_omp(int *a, int *b, int n, int pattern);
double testScan_ass(int* a, int* b, int n, int pattern);

double map_CPU(int *source, int *dest, int n);
double map_MIC(int *source, int *dest, int n, int k);

#endif