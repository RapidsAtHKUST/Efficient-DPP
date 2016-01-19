//
//  DataUtil.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__DataUtil__
#define __gpuqp_opencl__DataUtil__

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#include <string.h>
#include <sys/time.h>   //timing
#include <iostream>
#include <climits>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "dataDefinition.h"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT                        "/Users/Bryan/gpuqp_opencl/gpuqp_opencl/"
#define DATADIR                             "/Users/Bryan/gpuqp_opencl/gpuqp_opencl/data/"
#endif


//data information
#define MAX_DATA_SIZE 16000000              

//Error information
#define ERR_HOST_ALLOCATION                 "Failed to allocate the host memory."
#define ERR_WRITE_BUFFER                    "Failed to write to the buffer."
#define ERR_READ_BUFFER                     "Failed to read back the device memory."
#define ERR_SET_ARGUMENTS                   "Failed to set the arguments."
#define ERR_EXEC_KERNEL                     "Failed to execute the kernel."
#define ERR_LOCAL_MEM_OVERFLOW              "Local memory overflow "
#define ERR_COPY_BUFFER                     "Failed to copy the buffer."
#define ERR_RELEASE_MEM                     "Failed to release the device memory object."

//Function information
#define FUNC_BEGIN                          std::cout<<"------ Start "<<__FUNCTION__<<" ------"<<std::endl
#define FUNC_END                            std::cout<<"------ End of "<<__FUNCTION__<<" ------"<<std::endl<<std::endl
#define FUNC_CHECK(res)                     if (res==true)                                      \
                                                std::cout<<__FUNCTION__<<" pass!"<<std::endl;   \
                                            else                                                \
                                                std::cout<<__FUNCTION__<<" fail!"<<"\t\t\t\t Failed!"<<std::endl;

#define SHOW_TIME(time)                     std::cout<<"Kernel time for "<<__FUNCTION__<<" is "<<(time)<<" ms."<<std::endl
#define SHOW_TOTAL_TIME(time)               std::cout<<"Total time for "<<__FUNCTION__<<" is "<<(time)<<" ms."<<std::endl
#define SHOW_DATA_NUM(num)                  std::cout<<"Data number: "<<num<<std::endl
#define SHOW_TABLE_R_NUM(num)               std::cout<<"Table R number: "<<num<<std::endl
#define SHOW_TABLE_S_NUM(num)               std::cout<<"Table S number: "<<num<<std::endl
#define SHOW_PARALLEL(localSize,gridSize)   std::cout<<"Local size: "<<localSize<<"  Grid size: "<<gridSize<<std::endl
#define SHOW_CHECKING                       std::cout<<"Checking result..."<<std::endl
#define SHOW_JOIN_RESULT(res)               std::cout<<"Join result: "<<res<<std::endl

//executing information
#define  varName(x) #x
#define  printExecutingKernel(kernel)       std::cout<<"Executing kernel: " <<#kernel<<std::endl
#define  checkLocalMemOverflow(localMem)    if (localMem > MAX_LOCAL_MEM_SIZE)  \
                                            std::cerr<<ERR_LOCAL_MEM_OVERFLOW<<"( "<<localMem<<" > "<<MAX_LOCAL_MEM_SIZE<<" )"<<std::endl
//parallel condition
#define BLOCKSIZE   (512)
#define GRIDSIZE    (1024)
#define MAX_LOCAL_MEM_SIZE (47 * 1000)              //local memory size of 47KB

//auxiliary setting
#define SHUFFLE_TIME(TIME)  (TIME/2)

//the fixed records & int array read from the external file

double diffTime(struct timeval end, struct timeval start) ;

void recordRandom(Record *records, int length, int max = INT_MAX);
void recordRandom_Only(Record *records, int length,  int times);
void recordSorted(Record *records, int length, int max = INT_MAX);
void recordSorted_Only(Record *records, int length);

void intRandom(int *intArr, int length, int max);
void intRandom_Only(int *intArr, int length,  int times);

double calCPUTime(clock_t start, clock_t end);
void checkErr(cl_int status, const char* name);
void printbinary(const unsigned int val, int dis) ;

int compInt ( const void * p, const void * q);
int compRecordAsc ( const void * a, const void * b);
int compRecordDec ( const void * a, const void * b);

//generate the fixed record & int array and write to the external memory.
void generateFixedRecords(Record* fixedRecords, int length, bool write, char *file);
void generateFixedArray(int *fixedArray, int length, bool write, char *file);

//read from the external memory
void readFixedRecords(Record* fixedRecords, char *file, int& recordLength);
void readFixedArray(int* fixedArray, char *file, int & arrayLength);

//parameters processing
void processBool(const char *arg, bool& var);

//result checking ancillary functions
int floorOfPower2(int a);

#endif /* defined(__gpuqp_opencl__DataUtil__) */
