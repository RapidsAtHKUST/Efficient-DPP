//
//  scatterImpl.cpp
//  gpuqp_opencl
//
//  Created by Zhuohang Lai on 4/10/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#include "../util/Plat.h"
using namespace std;

double scatter(cl_mem d_in, cl_mem d_out, int length, cl_mem d_loc, int localSize, int gridSize, int pass) {
    device_param_t param = Plat::get_device_param();

    cl_event event;
    double totalTime = 0;
    cl_int status = 0;
    int argsNum = 0;

    //kernel reading
    cl_kernel scatterKernel = get_kernel(param.device, param.context, "scatter_kernel.cl", "scatter");

    //set kernel arguments
    int globalSize = gridSize * localSize;
    int ele_per_thread = (length + globalSize - 1) / globalSize;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    argsNum = 0;
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(cl_mem), &d_loc);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(scatterKernel, argsNum++, sizeof(int), &ele_per_thread);
    checkErr(status, ERR_SET_ARGUMENTS);

    //multi-pass kernel
    int len_per_run = (length + pass - 1) / pass;
    for(int i = 0; i < pass; i++) {
        int from = i * len_per_run;
        int to = (i+1) * len_per_run;
        status |= clSetKernelArg(scatterKernel, 5, sizeof(int), &from);
        status |= clSetKernelArg(scatterKernel, 6, sizeof(int), &to);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clFinish(param.queue);
        status = clEnqueueNDRangeKernel(param.queue, scatterKernel, 1, 0, global, local, 0, 0, &event);
        clFlush(param.queue);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);

        totalTime += clEventTime(event);
    }
    return totalTime;
}
