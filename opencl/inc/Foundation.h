//
//  Foundation.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/27/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef gpuqp_opencl_Foundation_h
#define gpuqp_opencl_Foundation_h

#include "KernelProcessor.h"
#include "PlatInit.h"

#include "CSSTree.h"
#include "dataDef.h"

//----------------------------- operations ------------------------
double vpu(
    cl_mem d_source_values, int length, 
    int localSize, int gridSize, PlatInfo& info, int con, int repeatTime, int basicSize) ;

double mem_read(
    cl_mem d_source_values, cl_mem d_dest_values, int length, 
    int localSize, int gridSize, PlatInfo& info, int con, int basicSize);

double mem_write(
    cl_mem d_source_values, int length, 
    int localSize, int gridSize, PlatInfo& info, int con, int basicSize);

double triad(
    cl_mem d_source_values_b, cl_mem d_source_values_c, cl_mem d_dest_values_a,int length, 
    int localSize, int gridSize, PlatInfo& info, int basicSize);

double my_barrier(
    cl_mem d_source_values, int localSize, int gridSize, PlatInfo& info, double& percentage);

double map(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    int localSize, int gridSize, PlatInfo& info) ;

double gather(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem &d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    cl_mem d_loc, int localSize, int gridSize, PlatInfo& info);

double scatter(
#ifdef RECORDS
    cl_mem d_source_keys, cl_mem &d_dest_keys, bool isRecord,
#endif
    cl_mem d_source_values, cl_mem& d_dest_values, int length, 
    cl_mem d_loc, int localSize, int gridSize, PlatInfo& info);


double scan(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize = BLOCKSIZE);
double scan_ble(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize = BLOCKSIZE);
DEPRECATED double scan_blelloch(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize = BLOCKSIZE);

double split(cl_mem d_source, cl_mem &d_dest, int length, int fanout, PlatInfo& info, int localSize, int gridSize);
double radixSort(cl_mem& d_source, int length, PlatInfo& info);

double bisort(cl_mem &d_source, int length, int dir, PlatInfo& info, int localSize, int gridSize);


double inlj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, CSS_Tree_Info treeInfo, PlatInfo info, int localSize, int gridSize);
double ninlj(cl_mem& d_R, int rLen, cl_mem& d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize, int gridSize);
double smj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize);
double partitionHJ(cl_mem& d_R, int rLen,int totalCountBits, PlatInfo info, int localSize, int gridSize) ;
double hj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int totalCountBits, int localSize);



//-------------------------test primitives-------------------------
void testVPU(
    float *fixedValues, 
    int length, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize) ;

void testMemRead(
    float *fixedValues, 
    int length, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

void testMemWrite(int length, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

void testTriad(
    float *fixedValues, 
    int length, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

void testBarrier(
    float *fixedValues, PlatInfo& info , double& totalTime, double& percentage, int localSize, int gridSize);

bool testMap(
#ifdef RECORDS
    int *fixedKeys,
#endif
    float *fixedValues, 
    int length, PlatInfo& info , double& totalTime, int localSize=BLOCKSIZE, int gridSize=GRIDSIZE);



bool testGather(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, int length, PlatInfo& info , double& totalTime, int localSize=BLOCKSIZE, int gridSize=GRIDSIZE);

bool testScatter(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, int length, PlatInfo& info , double& totalTime, int localSize=BLOCKSIZE, int gridSize=GRIDSIZE);

bool testRadixSort(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, 
    int length, PlatInfo& info, double& totalTime);

bool testScan(int *fixedSource, int length, PlatInfo& info, double& totalTime, int isExclusive, int localSize = BLOCKSIZE);
bool testSplit(Record *fixedSource, int length, PlatInfo& info , int fanout, double& totalTime, int localSize= BLOCKSIZE, int gridSize = GRIDSIZE);
bool testBitonitSort(Record *fixedSource, int length, PlatInfo& info, int dir, double& totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);

//-------------------------test joins-------------------------
bool testNinlj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testInlj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testSmj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE);
bool testHj(int rLen, int sLen, PlatInfo& info, int countBit, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
#endif