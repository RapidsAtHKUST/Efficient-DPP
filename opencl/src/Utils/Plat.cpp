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
    cl_device_id devices[MAX_DEVICES_NUM];

    /*get platforms*/
    status = clGetPlatformIDs(0, 0, &plat_num);    //check number of platforms
    checkErr(status,"No platform available.");

    /*only 1 platform used*/
    status = clGetPlatformIDs(MAX_PLATFORM_NUM, &this->platform, NULL);
    memset(platform_name, '\0', sizeof(char)*200);
    status = clGetPlatformInfo(this->platform, CL_PLATFORM_NAME, 200, platform_name, NULL);
    cout<<"Platform: "<<platform_name<<endl;

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

//    cin >> chosenDevice;
    this->chosen_device_id = 1;
    if (this->chosen_device_id < 0 || this->chosen_device_id >= num_devices)   {
        cerr<<"Wrong parameter."<<endl;
        exit(1);
    }
    cl_device_id my_device = devices[this->chosen_device_id];
    cout<<"Selected device: "<<devices_name[this->chosen_device_id]<<endl;
    this->device_params.device = my_device;

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
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulong), &this->device_params.gmem_size, NULL);
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(size_t), &this->device_params.cacheline_size, NULL);

    std::cout<<"------ End of hardware checking ------"<<endl<<endl;
}

cl_kernel Plat::get_kernel(char *file_name, char *func_name, char *params) {

    device_param_t param = Plat::instance->device_params;

/*read the raw kernel file*/
    char *addr = new char[200];
    strcat(addr, PROJECT_ROOT);
    strcat(addr, "/Kernels/");
    strcat(addr, file_name);

    ifstream in(addr,std::fstream::in| std::fstream::binary);
    if(!in.good()) {
        cerr<<"Kernel file not exist!"<<endl;
        exit(1);
    }
    
    /*get file length*/
    in.seekg(0, std::ios_base::end);    //jump to the end
    size_t length = in.tellg();         //read the length
    in.seekg(0, std::ios_base::beg);    //jump to the front
    
    //read program source
    char *source = new char[length+1];
    in.read(source, length);            //read the kernel file
    source[length] = '\0';              //set the last one char to '\0'

/*compile the kernel file*/
    cl_int status;
    cl_program program = clCreateProgramWithSource(param.context, 1, (const char**)(&source), 0, &status);
    checkErr(status, "Failed to creat program.");

    char *args = new char[1000];
    strcat(args, "-I");
    strcat(args, PROJECT_ROOT);
    strcat(args, "/inc ");

    strcat(args," -DKERNEL ");
    strcat(args, params);

    // strcat(args, " -auto-prefetch-level=0 ");    
    status = clBuildProgram(program, 1, &param.device, args, 0, 0);
    checkErr(status, "Compilation error.");

/*create the kernel*/
    cl_kernel kernel = clCreateKernel(program, func_name, &status);
    checkErr(status, "Kernel function name not found.");

    if(source)  delete[] source;
    if(addr)    delete[] addr;
    if(args)    delete[] args;

    return kernel;
}

/*deconstruction*/
Plat::~Plat() {
    clReleaseContext(this->device_params.context);
    clReleaseCommandQueue(this->device_params.queue);

    if (Plat::instance) {
        instance = NULL;
    }
}