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
#include "DataDef.h"

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

double gather(cl_mem d_source_values, cl_mem& d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, const PlatInfo info, int pass);

double scatter(cl_mem d_source_values, cl_mem& d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, const PlatInfo info, int pass);

double scan_fast(cl_mem &d_inout, int length, PlatInfo& info, int localSize, int gridSize, int R, int L);
double scan_three_kernel(cl_mem &d_inout, unsigned length, PlatInfo &info, int local_size, int grid_size);
double scan_three_kernel_single(cl_mem &d_inout, unsigned length, PlatInfo &info, int grid_size);

double scan(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize = BLOCKSIZE);


DEPRECATED double scan_blelloch(cl_mem &cl_arr, int num,int isExclusive, PlatInfo& info, int localSize = BLOCKSIZE);

/*split algorithms*/
double block_split(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, bool reorder, PlatInfo& info, cl_mem d_in_values=NULL, cl_mem d_out_values=NULL, int local_size=256, int grid_size=32768);

double thread_split(cl_mem d_in_keys, cl_mem d_out_keys, cl_mem d_start, int length, int buckets, PlatInfo& info, cl_mem d_in_values=NULL, cl_mem d_out_values=NULL, int local_size=256, int grid_size=32768);

double single_split(cl_mem d_in_keys, cl_mem d_out_keys, int length, int buckets, bool reorder, PlatInfo& info, cl_mem d_in_values=NULL, cl_mem d_out_values=NULL);

/*end of split algorithms*/

double radixSort(cl_mem& d_source, int length, PlatInfo& info);

double bisort(cl_mem &d_source, int length, int dir, PlatInfo& info, int localSize, int gridSize);


double inlj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, CSS_Tree_Info treeInfo, PlatInfo info, int localSize, int gridSize);
double ninlj(cl_mem& d_R, int rLen, cl_mem& d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize, int gridSize);
double smj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, PlatInfo info, int localSize);
double partitionHJ(cl_mem& d_R, int rLen,int totalCountBits, PlatInfo info, int localSize, int gridSize) ;
double hashjoin(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len, PlatInfo info);
double hashjoin_np(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len, PlatInfo info);



//-------------------------test primitives-------------------------
void testMem(PlatInfo& info);
void test_wg_sequence(unsigned long len, PlatInfo& info);
void testAccess(PlatInfo& info);
bool testGather(int len, const PlatInfo info);
bool testScatter(int len, const PlatInfo info);
void testAtomic(PlatInfo& info);
bool testScan(int length, double &totalTime, int localSize, int gridSize, int R, int L, PlatInfo& info);
void testScanParameters(int length, int selection, PlatInfo& info);
bool testSplit(int len, PlatInfo& info, int buckets, double& totalTime, int test_type);
void testSplitParameters(int len, int buckets, int device, int algo, PlatInfo& info);





void testBarrier(
    float *fixedValues, PlatInfo& info , double& totalTime, double& percentage, int localSize, int gridSize);



void testLatency(PlatInfo& info);

bool testMap(PlatInfo& info, int repeat, int repeatTrans, int localSize=BLOCKSIZE, int gridSize=GRIDSIZE);



bool testRadixSort(
#ifdef RECORDS
    int *fixedKeys,
#endif
    int *fixedValues, 
    int length, PlatInfo& info, double& totalTime);


bool testBitonitSort(Record *fixedSource, int length, PlatInfo& info, int dir, double& totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);

//-------------------------test joins-------------------------
bool testNinlj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testInlj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE, int gridSize = GRIDSIZE);
bool testSmj(int rLen, int sLen, PlatInfo& info, double &totalTime, int localSize = BLOCKSIZE);
bool testHj(int rLen, int sLen, PlatInfo info);
#endif