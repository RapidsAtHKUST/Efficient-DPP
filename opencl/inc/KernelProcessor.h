//
//  KernelReader.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__KernelReader__
#define __gpuqp_opencl__KernelReader__

#include <stdio.h>
#include <string>
#include "DataUtil.h"

class KernelProcessor {
private:
    cl_program program;
    char *source;
    cl_kernel &createKernel(char *kerName);
protected:
    void kernelRead(char *addr);
    void compile(cl_context context, char* extra);
public:
    KernelProcessor(char *addr, cl_context context, char* extra="");
    static cl_kernel &getKernel(char* fileName, char* funcName, cl_context context, char* flags = "");
    ~KernelProcessor();
};

#endif /* defined(__gpuqp_opencl__KernelReader__) */
