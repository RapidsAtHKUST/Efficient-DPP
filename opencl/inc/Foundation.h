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

double map_hashing(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat) ;

double map_branching(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat, int branch) ;

double map_branching_for(
    cl_mem d_source_values, cl_mem& d_dest_values, int localSize, int gridSize, PlatInfo& info, int repeat, int branch) ;

template<typename T>
void map_transform(
    cl_mem d_source_alpha, cl_mem& d_source_beta, T r, 
    cl_mem &d_dest_x, cl_mem& d_dest_y, cl_mem& d_dest_z,
    int localSize, int gridSize, PlatInfo& info, int repeat, double &blank_time, double &total_time);

double gather(cl_mem d_source_values, cl_mem& d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, const PlatInfo info, int numOfRun);

double scatter(cl_mem d_source_values, cl_mem& d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, const PlatInfo info, int numOfRun);


double scan_fast(cl_mem &d_source, int length, int isExclusive, PlatInfo& info, int localSize, int gridSize, int R, int L);

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
template<typename T>
void testVPU(T *fixedValues, PlatInfo& info , double& totalTime, int localSize, int gridSize, int basicSize);

template<typename T>
void testMem(PlatInfo& info , const int blockSize, const int gridSize, double& readTime, double& writeTime, double& mulTime, double& addTime, int repeat);

template<typename T>
void testAccess(PlatInfo& info , const int blockSize, const int gridSize, int repeat);

void testBarrier(
    float *fixedValues, PlatInfo& info , double& totalTime, double& percentage, int localSize, int gridSize);

void testAtomic(PlatInfo& info , double& totalTime, int localSize, int gridSize, bool isLocal);

void testLatency(PlatInfo& info);

bool testMap(PlatInfo& info, int repeat, int repeatTrans, int localSize=BLOCKSIZE, int gridSize=GRIDSIZE);


bool testGather(int *fixedValues, const int lengthMax, const PlatInfo info);

bool testScatter(int *fixedValues, const int lengthMax, const PlatInfo info);

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