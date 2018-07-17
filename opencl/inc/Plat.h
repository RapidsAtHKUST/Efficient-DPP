//
//  Plat.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__Plat__
#define __gpuqp_opencl__Plat__

#include "general.h"

#define MAX_PLATFORM_NUM 1          /*only 1 platform available*/
#define MAX_DEVICES_NUM 10          /*at most 10 device can be detected*/

typedef struct device_param_t {
    /*OpenCL constructs*/
    cl_platform_id      platform;           /*current platform*/
    cl_device_id        device;             /*current device*/
    char                device_name[200];   /*name of the device*/
    cl_context          context;            /*current context*/
    cl_command_queue    queue;              /*current command queue*/

    /*hardware properties*/
    uint64_t            gmem_size;          /*global memory size*/
    uint64_t            cacheline_size;     /*global memory cache line size*/
    uint64_t            lmem_size;          /*local memory size*/
    uint64_t            cus;                /*number of CUs*/
    uint64_t            max_alloc_size;     /*maximal memory object alloc size*/
    uint64_t            max_local_size;     /*maximal local size*/
    uint64_t            wavefront;          /*wavefront size*/
} device_param_t;

/*
 * Platform class, used to initialize the device and set up the device_param.
 * */
class Plat
{
private:
    cl_platform_id platform;                            /*current platform*/
    cl_device_type type;                                /*device type*/
    uint num_devices;                                   /*number of devices*/
    uint chosen_device_id;                              /*the device id chosen*/
    device_param_t device_params;                       /*current device parameters*/
    static Plat *instance;                              /*singleton instance*/

    void init_properties();                             /*init the hardware properties*/
protected:
    Plat();
    static Plat *getInstance();                         /*get the singleton*/
    static void autoDestroy();                          /*only for auto call ~Plat()*/
    ~Plat();                                            /*deconstructor*/
public:
    static void plat_init(cl_device_type type=CL_DEVICE_TYPE_ALL);    /*initialize the platform with device_type and choose the device*/
    static device_param_t get_device_param();      /*get params of current device*/


};
#endif
