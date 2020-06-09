//
//  gatherImpl.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "../util/Plat.h"

double gather(cl_mem d_in, cl_mem d_out, int length, cl_mem d_loc, int localSize, int gridSize, int pass) {
    device_param_t param = Plat::get_device_param();

    cl_event event;
    double totalTime = 0;
    cl_int status = 0;
    int argsNum = 0;
    
    //kernel reading
    cl_kernel gatherKernel = get_kernel(param.device, param.context, "gather_kernel.cl", "gather");

    //set kernel arguments
    int globalSize = gridSize * localSize;
    int ele_per_thread = (length + globalSize - 1) / globalSize;

    //set work group and NDRange sizes
    size_t local[1] = {(size_t)localSize};
    size_t global[1] = {(size_t)(localSize * gridSize)};

    argsNum = 0;
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_in);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_out);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(cl_mem), &d_loc);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(int), &length);
    status |= clSetKernelArg(gatherKernel, argsNum++, sizeof(int), &ele_per_thread);
    checkErr(status, ERR_SET_ARGUMENTS);

    //multi-pass kernel
    int len_per_run = (length + pass - 1) / pass;
    for(int i = 0; i < pass; i++) {
        int from = i * len_per_run;
        int to = (i+1) * len_per_run;
        status |= clSetKernelArg(gatherKernel, 5, sizeof(int), &from);
        status |= clSetKernelArg(gatherKernel, 6, sizeof(int), &to);
        checkErr(status, ERR_SET_ARGUMENTS);

        status = clFinish(param.queue);
        status = clEnqueueNDRangeKernel(param.queue, gatherKernel, 1, 0, global, local, 0, 0, &event);
        clFlush(param.queue);
        status = clFinish(param.queue);
        checkErr(status, ERR_EXEC_KERNEL);

        totalTime += clEventTime(event);
    }
    return totalTime;
}
