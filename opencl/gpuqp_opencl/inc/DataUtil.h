//
//  DataUtil.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__DataUtil__
#define __gpuqp_opencl__DataUtil__

#include "utility.h"


#ifndef PROJECT_ROOT
#define PROJECT_ROOT                        "/Users/Bryan/gpuqp_opencl/gpuqp_opencl/"
#endif

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

#define MAX_LOCAL_MEM_SIZE (47 * 1000)              //local memory size of 47KB
                                            

#endif /* defined(__gpuqp_opencl__DataUtil__) */
