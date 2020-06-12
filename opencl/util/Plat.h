//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//

#pragma once

#include <fstream>
#include <iostream>
#include "../primitives.h"
using namespace std;

#define MAX_PLATFORM_NUM 1          /*only 1 platform available*/
#define MAX_DEVICES_NUM 10          /*at most 10 device can be detected*/

struct device_param_t {
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
};

/*
 * Platform class, used to initialize the device and set up the device_param.
 * */
class Plat {
private:
    cl_platform_id _platform;                            /*current platform*/
    cl_device_type _type;                                /*device type*/
    uint _num_devices;                                   /*number of devices*/
    uint _chosen_device_id;                              /*the device id chosen*/
    device_param_t _device_params;                       /*current device parameters*/
    static Plat *_instance;                              /*singleton instance*/

    void init_properties();                             /*init the hardware properties*/
protected:
    Plat() {};
    static Plat *getInstance();                         /*get the singleton*/
    static void autoDestroy();                          /*only for auto call ~Plat()*/
    ~Plat() {
        clReleaseContext(this->_device_params.context);
        clReleaseCommandQueue(this->_device_params.queue);

        if (Plat::_instance) {
            _instance = nullptr;
        }
    };                                            /*deconstructor*/
public:
    static void plat_init(cl_device_type type=CL_DEVICE_TYPE_ALL);    /*initialize the platform with device_type and choose the device*/
    static device_param_t get_device_param();      /*get params of current device*/
};