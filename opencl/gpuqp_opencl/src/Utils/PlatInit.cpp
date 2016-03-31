//
//  PlatInit.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "PlatInit.h"
#include <iostream>
using namespace std;

PlatInit::PlatInit() {}
PlatInit::PlatInit(const PlatInit&) {}

void PlatInit::autoDestroy() {
    if (instance != NULL) {
         delete instance;
    }
}

PlatInit* PlatInit::instance = NULL;

PlatInit* PlatInit::getInstance(int gpu) {
    if (1 != gpu && 0 != gpu && 2 != gpu && 3 != gpu) {     //check the correctness of the para
        cerr<<"Wrong parameter for PlatInit::getInstance"<<endl;
        return NULL;
    }

    if (PlatInit::instance == NULL) {
        instance = new PlatInit();
        instance->gpu = gpu;
        
        cout<<endl;
        if (gpu == 0)           cout<<"Detection mode: All"<<endl;
        else if (gpu == 1)      cout<<"Detection mode: GPUs only"<<endl;
        else if (gpu == 2)      cout<<"Detection mode: CPUs only"<<endl;
        else                    cout<<"Detection mode: Accelerators only"<<endl;
        
        instance->initPlat();
        atexit(autoDestroy);                        //to call destroy() before exit
    }

    return instance;
}

void PlatInit::initPlat() {
    
    std::cout<<"------ Start hardware checking ------"<<endl;
    cl_int status;                                 //func return value
    cl_uint num;                                    //loop num
    char dname[MAX_DEVICES_NUM][1500];              //info show, at most 10 devices can be detected
    
    int chosenDevice = -1;                      //the index of the chosen device
    cl_device_id chosenDeviceID;
    
    status = clGetPlatformIDs(0, 0, &num);         //check number of platforms
    checkErr(status,"No platform available.");
    
    status = clGetPlatformIDs(1, &this->platform, NULL);
    for(int i = 0;i<num;i++) {
        status = clGetPlatformInfo(this->platform, CL_PLATFORM_NAME, 1500, dname[0], NULL);
        cout<<"Platform: "<<dname[0]<<endl;
    }
    
    //decide the detection mode
    cl_device_type type;
    if (this->gpu == 0)             type = CL_DEVICE_TYPE_ALL;
    else if (this->gpu == 1)        type = CL_DEVICE_TYPE_GPU;
    else if(this->gpu == 2)         type = CL_DEVICE_TYPE_CPU;
    else                            type = CL_DEVICE_TYPE_ACCELERATOR;
    
    //step 1: get device ids
    status = clGetDeviceIDs(this->platform, type, 0, 0, &this->numOfDev);
    checkErr(status, "No devices available");
    status = clGetDeviceIDs(this->platform, type, this->numOfDev, devices, NULL);
    cout<<"Number of devices: "<<this->numOfDev<<endl;
    
    for(int i = 0;i<this->numOfDev;i++) {
        status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1500, dname[i], NULL);
        cout<<"\tComputing device " <<i<<" : "<<dname[i]<<endl;
    }
    
    cout<<"Please enter the index of the device to use (0,1,2...) : ";
    // cin >> chosenDevice;
    chosenDevice = 1;   //mic
    if (chosenDevice < 0 || chosenDevice >= numOfDev)   {
        cerr<<"Wrong parameter."<<endl;
        exit(1);
    }
    
    cout<<"Selected device: "<<dname[chosenDevice]<<endl;
    chosenDeviceID = devices[chosenDevice];
    
    //step 2 : create the context
    const cl_context_properties prop[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(this->platform),0};
    
    //only 1 device is used
    this->context = clCreateContext(prop, 1, &devices[chosenDevice], NULL, NULL, &status);
    checkErr(status, "Fail to create the context.");
    
    //step 3 : create the command queue
    this->queue = clCreateCommandQueue(context, devices[chosenDevice], 0, &status);
    checkErr(status, "Failed to create the command queue.");
    
    std::cout<<"------ End of hardware checking ------"<<endl<<endl;
}

cl_command_queue PlatInit::getQueue() {
    return this->queue;
}

void PlatInit::setGPU(int gpu) {
    if (1 != gpu && 0 != gpu && 2 != gpu && 3 != gpu) {
        cerr<<"Wrong parameter for PlatInit::getInstance."<<endl;
        return;
    }
    PlatInit::gpu = gpu;
    initPlat();
}

unsigned int PlatInit::getNumOfDev() {
    return this->numOfDev;
}

cl_context PlatInit::getContext() {
    return this->context;
}

cl_device_id* PlatInit::getDevices() {
    return this->devices;
}

PlatInit::~PlatInit() {

    clReleaseCommandQueue(queue);

    clReleaseContext(context);
    
    if (PlatInit::instance) {
        instance = NULL;
    }
}

PlatInfo getInfo() {
    //platform initialization
    PlatInit* myPlatform = PlatInit::getInstance();
    cl_command_queue queue = myPlatform->getQueue();
    cl_context context = myPlatform->getContext();
    cl_command_queue currentQueue = queue;
    
    PlatInfo info;
    info.context = context;
    info.currentQueue = currentQueue;
    
    return info;
}



