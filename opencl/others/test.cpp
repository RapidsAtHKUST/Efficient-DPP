#include <iostream>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/OpenCL.h>
#else
#include "CL/cl.h"
#endif

#include "Foundation.h"

void testMem(PlatInfo& info) {
    cl_int status = 0;
    cl_event event;
    int argsNum;
    int localSize = 1024;
    int gridSize = 262144;
    double mulTime = 0.0;
    int scalar = 13;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)(localSize)};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    //get the kernel
    cl_kernel mul_kernel = KernelProcessor::getKernel("memKernel.cl", "mem_mul_bandwidth", info.context);

    int len = localSize*gridSize;
    std::cout<<"Data size for read/write(multiplication test): "<<len<<" ("<<len*sizeof(int)*1.0/1024/1024<<"MB)"<<std::endl;

    //data initialization
    int *h_in = new int[len];
    for(int i = 0; i < len; i++) h_in[i] = i;
    cl_mem d_in = clCreateBuffer(info.context, CL_MEM_READ_WRITE , sizeof(int)*len, NULL, &status);
    cl_mem d_out = clCreateBuffer(info.context, CL_MEM_WRITE_ONLY , sizeof(int)*len, NULL, &status);
    checkErr(status, ERR_HOST_ALLOCATION);
    status = clEnqueueWriteBuffer(info.currentQueue, d_in, CL_TRUE, 0, sizeof(int)*len, h_in, 0, 0, 0);
    checkErr(status, ERR_WRITE_BUFFER);

    //set kernel arguments
    argsNum = 0;
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(mul_kernel, argsNum++, sizeof(int), &scalar);
    checkErr(status, ERR_SET_ARGUMENTS);

    status = clFinish(info.currentQueue);
    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        status = clEnqueueNDRangeKernel(info.currentQueue, mul_kernel, 1, 0, global, local, 0, 0, &event);  //kernel execution
        status = clFinish(info.currentQueue);
        checkErr(status, ERR_EXEC_KERNEL);

        //throw away the first result
        if (i != 0) mulTime += clEventTime(event);
    }
    mulTime /= (MEM_EXPR_TIME - 1);

    status = clFinish(info.currentQueue);
    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_out);
    checkErr(status, ERR_RELEASE_MEM);
    delete[] h_in;

    //compute the bandwidth, including read and write
    double throughput = computeMem(len*2, sizeof(int), mulTime);
    cout<<"Time for multiplication: "<<mulTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput<<" GB/s"<<endl;
}

int main() {
    PlatInit* myPlatform = PlatInit::getInstance(0);
    cl_command_queue queue = myPlatform->getQueue();
    cl_context context = myPlatform->getContext();
    cl_command_queue currentQueue = queue;

    info.context = context;
    info.currentQueue = currentQueue;

    testMem(info.context);

    return 0;
}