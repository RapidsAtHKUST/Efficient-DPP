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
    char **source;
    int num;
protected:
    void kernelRead(std::string *addr, int num);
    void compile(cl_context context);
public:
    KernelProcessor(std::string *addr,int num, cl_context context);
    cl_kernel &getKernel(char *kerName);
    ~KernelProcessor();
};

#endif /* defined(__gpuqp_opencl__KernelReader__) */
