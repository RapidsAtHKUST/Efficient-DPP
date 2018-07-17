//
//  PlatInit.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Plat.h"
#include <fstream>
#include <iostream>
using namespace std;

Plat* Plat::instance = NULL;

Plat::Plat() {}

void Plat::autoDestroy() {
    if (instance != NULL) {
        delete instance;
    }
}

void Plat::plat_init(cl_device_type new_type) {
    if (Plat::instance == NULL) {
        cout<<endl<<"Device type: ";
        if (new_type == CL_DEVICE_TYPE_ALL)         cout<<"All"<<endl;
        else if (new_type == CL_DEVICE_TYPE_GPU)    cout <<"GPUs only"<<endl;
        else if (new_type == CL_DEVICE_TYPE_CPU)    cout <<"CPUs only"<<endl;
        else if (new_type == CL_DEVICE_TYPE_ACCELERATOR)    cout <<"Accelerators only"<<endl;
        else {
            cerr<<"Wrong device type."<<endl;
            return;
        }

        /*initilize the plaform*/
        Plat::instance = new Plat();
        instance->type = new_type;
        instance->init_properties();
        atexit(autoDestroy);               /*to call destroy() before exit*/
    }
    else {
        std::cout<<"Platform has been initialized"<<std::endl;
    }
}

device_param_t Plat::get_device_param() {
    if (Plat::instance == NULL) {
        cerr<<"Platform and Deivce have not been initialized."<<endl;
        exit(1);
    }
    else {
        uint idx = Plat::instance->chosen_device_id;
        return Plat::instance->device_params;
    }
}

void Plat::init_properties() {
    cout<<"------ Start hardware checking ------"<<endl;
    cl_int status;
    cl_uint plat_num;
    char platform_name[200];                        /*platform name*/
    char devices_name[MAX_DEVICES_NUM][200];        /*devices name*/
    char cl_version_info[100];
    cl_device_id devices[MAX_DEVICES_NUM];

    /*get platforms*/
    status = clGetPlatformIDs(0, 0, &plat_num);    //check number of platforms
    checkErr(status,"No platform available.");

    /*only 1 platform used*/
    status = clGetPlatformIDs(MAX_PLATFORM_NUM, &this->platform, NULL);
    memset(platform_name, '\0', sizeof(char)*200);
    status = clGetPlatformInfo(this->platform, CL_PLATFORM_NAME, 200, platform_name, NULL);
    cout<<"Platform: "<<platform_name<<endl;
    this->device_params.platform = this->platform;

    /*get device IDs*/
    status = clGetDeviceIDs(this->platform, this->type, 0, 0, &this->num_devices);
    checkErr(status, "No devices available");
    status = clGetDeviceIDs(this->platform, this->type, this->num_devices, devices, NULL);
    cout<<"Number of devices: "<<this->num_devices<<endl;

    for(int i = 0; i< this->num_devices; i++) {
        memset(devices_name[i], '\0', sizeof(char)*200);
        status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 200, devices_name[i], NULL);
        cout<<"\tComputing device " <<i<<" : "<<devices_name[i]<<endl;
    }

    cout<<"Please enter the index of the device to use (0,1,2...) : ";

    cin >> this->chosen_device_id;
//    this->chosen_device_id = 1;
    if (this->chosen_device_id < 0 || this->chosen_device_id >= num_devices)   {
        cerr<<"Wrong parameter."<<endl;
        exit(1);
    }
    cl_device_id my_device = devices[this->chosen_device_id];
    cout<<"Selected device: "<<devices_name[this->chosen_device_id]<<endl;
    this->device_params.device = my_device;

    clGetDeviceInfo(my_device, CL_DEVICE_VERSION, sizeof(char)*100, cl_version_info, NULL); /*retrieve the OpenCL support version*/

    /*create the context*/
    const cl_context_properties prop[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(this->platform),0};
    cl_context my_context = clCreateContext(prop, 1, &my_device, NULL, NULL, &status);
    checkErr(status, "Fail to create the context."); //only 1 device is used
    this->device_params.context = my_context;

    /*create the command queue*/
    cl_command_queue my_queue = clCreateCommandQueue(my_context, my_device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkErr(status, "Failed to create the command queue.");
    this->device_params.queue = my_queue;

    /*initialize other params*/
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &this->device_params.gmem_size, NULL);  /*global memory size*/
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(uint64_t), &this->device_params.cacheline_size, NULL); /*cacheline size*/
    clGetDeviceInfo(my_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &this->device_params.lmem_size, NULL); /*local memory size*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint64_t), &this->device_params.cus, NULL);       /*number of CUs*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(uint64_t), &this->device_params.max_alloc_size, NULL);       /*number of CUs*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &this->device_params.max_local_size, NULL);       /*maximal local size*/

    /*get the wavefront size according to the device type*/
    cl_device_type my_type;
    clGetDeviceInfo(my_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &my_type, NULL);

    if (my_type == CL_DEVICE_TYPE_GPU) {    /*GPUs*/
        /*a simple kernel*/
        cl_kernel temp_kernel = get_kernel(this->device_params.device, this->device_params.context, "gather_kernel.cl", "gather");
        clGetKernelWorkGroupInfo(temp_kernel, this->device_params.device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(uint64_t), &this->device_params.wavefront, NULL);
    }
    else {      /*CPUs and MICs*/
        clGetDeviceInfo(this->device_params.device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(uint64_t), &this->device_params.wavefront, NULL);
    }

    /*display the params*/
    cout<<"\tVersion: "<<cl_version_info<<endl;
    cout<<"\tGlobal memory size: "<<this->device_params.gmem_size*1.0/1024/1024/1024<<" GB"<<endl;
    cout<<"\tLocal memory size: "<<this->device_params.lmem_size*1.0/1024<<" KB"<<endl;
    cout<<"\tCompute units: "<<this->device_params.cus<<endl;
    cout<<"\tMaximal local_size: "<<this->device_params.max_local_size<<endl;
    cout<<"\tMaximal memory object size: "<<this->device_params.max_alloc_size*1.0/1024/1024/1024<<" GB"<<endl;
    cout<<"\tGlobal memory cache line size: "<<this->device_params.cacheline_size<<" Byte"<<endl;
    cout<<"\tWavefront size: "<<this->device_params.wavefront<<endl;

    std::cout<<"------ End of hardware checking ------"<<endl<<endl;
}

/*deconstruction*/
Plat::~Plat() {
    clReleaseContext(this->device_params.context);
    clReleaseCommandQueue(this->device_params.queue);

    if (Plat::instance) {
        instance = NULL;
    }
}