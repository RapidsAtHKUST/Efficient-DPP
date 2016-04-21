//
//  KernelReader.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "KernelProcessor.h"
#include <fstream>
#include <iostream>
using namespace std;

KernelProcessor::KernelProcessor(string *addr,int num, cl_context context) {
    this->num = num;
    source = new char*[this->num];
    kernelRead(addr, num);
    compile(context);
}

void KernelProcessor::kernelRead(string *addr, int num) {
    for(int i = 0; i < num; ++i ) {
        ifstream in(addr[i].c_str(),std::fstream::in| std::fstream::binary);
        
        if(!in.good()) {
            cerr<<"Kernel file not exist!"<<endl;
            exit(1);
        }
        
        //get file length
        in.seekg(0, std::ios_base::end);    //jump to the end
        size_t length = in.tellg();         //read the length
        in.seekg(0, std::ios_base::beg);    //jump to the front
        
        //read program source
        source[i] = new char[length+1];
        
        in.read(source[i], length);            //read the kernel file
        source[i][length] = '\0';              //set the last one char to '\0'
    }
}

void KernelProcessor::compile(cl_context context) {
    //establish the program the compile it
    cl_int err;
    this->program = clCreateProgramWithSource(context, this->num, (const char**)this->source, 0, &err);
    checkErr(err, "Failed to creat program.");
    
    size_t totalSize;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &totalSize);
    
    long num = totalSize/sizeof(cl_device_id);
    cl_device_id devices[num];
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, totalSize, devices, NULL);
    checkErr(err, "Failed to get the devices of the context.");
    
    //compile
    char path[1000] = "-I";
    strcat(path, PROJECT_ROOT);
    strcat(path, "/inc ");

    // strcat(path, " -I");
    // strcat(path, PROJECT_ROOT);
    // strcat(path, "/common ");
    strcat(path,"-DKERNEL ");
#ifdef RECORDS
    strcat(path,"-DRECORDS");
#endif
    
    cout<<"building programs"<<endl;
    err = clBuildProgram(program, cl_int(num), devices,path, 0, 0);
    checkErr(err, "Compilation error.");
    cout<<"building programs complete"<<endl;

    
}

cl_kernel &KernelProcessor::getKernel(char* kerName) {
    //extract kernel
    cl_int err;
    cl_kernel *kernel = new cl_kernel;
    *kernel = clCreateKernel(this->program, kerName, &err);
    checkErr(err, "Kernel function name not found.");
    return *kernel;
}

KernelProcessor::~KernelProcessor() {
    delete source;
    source = NULL;
}



