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

KernelProcessor::KernelProcessor(string *addr,int num, cl_context context, char* extra) {
    this->num = num;
    source = new char*[this->num];
    kernelRead(addr, num);
    compile(context, extra);
}

void KernelProcessor::kernelRead(string *addr, int num) {
//    std::cout<<"Kernel Address: "<<*addr<<std::endl;
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

void KernelProcessor::compile(cl_context context, char* extra) {

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
    char comArgs[1000] = "-I";
    strcat(comArgs, PROJECT_ROOT);
    strcat(comArgs, "/inc ");

    strcat(comArgs,"-DKERNEL ");
    strcat(comArgs, extra);

    // strcat(comArgs, " -auto-prefetch-level=0 ");    
    
#ifdef RECORDS
    strcat(comArgs,"-DRECORDS");
#endif
//    std::cout<<"Compile kernel with flags: "<<comArgs<<std::endl;
    err = clBuildProgram(program, cl_int(num), devices,comArgs, 0, 0);
    checkErr(err, "Compilation error.");

    // //extract the assembly programs
    // size_t ass_size;
    // err = clGetProgramInfo(this->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &ass_size, NULL);
    // checkErr(err,"Failed to get the size of the assembly program.");

    // unsigned char *binary = new unsigned char[ass_size];
    // err = clGetProgramInfo(this->program, CL_PROGRAM_BINARIES, ass_size, &binary, NULL);
    // checkErr(err,"Failed to generate the assembly program.");

    // FILE * fpbin = fopen( "assembly.ass", "wb" );
    // if( fpbin == NULL )
    // {
    //     fprintf( stderr, "Cannot create '%s'\n", "assembly.ass" );
    // }
    // else
    // {
    //     fwrite( binary, 1, ass_size, fpbin );
    //     fclose( fpbin );
    // }
    // delete [] binary;

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



