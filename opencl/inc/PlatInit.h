//
//  Plat.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__Plat__
#define __gpuqp_opencl__Plat__

#include "DataUtil.h"
#define  MAX_DEVICES_NUM 10         //at most 10 device can be detected

typedef struct PlatInfo {
    cl_device_id device;
    cl_context context;
    cl_command_queue currentQueue;
} PlatInfo;

PlatInfo getInfo();

class Plat
{
private:

    int gpu;                                    //whether to detect gpu only, cpu only or all the devices :
                                                //0 -- use all
                                                //1 -- use gpu only
                                                //2 -- use cpu only
                                                //3 -- use accelerator only (e.g:MIC)
    
    unsigned int numOfDev;                      //number of devices
    cl_platform_id platform;                    //current platform id
    cl_device_id devices[MAX_DEVICES_NUM];      //devices id array
    cl_device_id device;                        //chosen device
    cl_context context;                        //current context
    cl_command_queue queue;                     //current command queue
    
protected:
    Plat();                                 //protected constructor
    Plat(const Plat&);                  //protected copy constructor
    Plat& operator=(const Plat&);       //protected assignment operator

    void initPlat();                            //platform initialization
    static Plat* instance;                  //singleton instance
    static void autoDestroy();                  //only for auto call ~Plat()
    ~Plat();                                //protected deconstructor
    
public:
    static Plat* getInstance(int gpu = 0);  //it detects all the devices by default
    cl_command_queue getQueue();
    void setGPU(int gpu);
    unsigned int getNumOfDev();
    cl_context getContext();
    cl_device_id getDevice();
    cl_device_id* getDevices();
};

#endif
