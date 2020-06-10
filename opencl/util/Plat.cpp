//
// Created by Zhuohang Lai on 9/6/2020.
//

#include "Plat.h"
#include "log.h"

Plat* Plat::_instance = nullptr;

void Plat::autoDestroy() {
    if (_instance != nullptr) {
        delete _instance;
    }
}

void Plat::plat_init(cl_device_type new_type) {
    if (Plat::_instance == nullptr) {
        if (new_type == CL_DEVICE_TYPE_ALL) log_info("Device type: All");
        else if (new_type == CL_DEVICE_TYPE_GPU)    log_info("Device type: GPUs only");
        else if (new_type == CL_DEVICE_TYPE_CPU)    log_info("Device type: CPUs only");
        else if (new_type == CL_DEVICE_TYPE_ACCELERATOR)    log_info("Device type: Accelerators only");
        else {
            log_error("Wrong device type");
            return;
        }

        /*initilize the plaform*/
        Plat::_instance = new Plat();
        _instance->_type = new_type;
        _instance->init_properties();
        atexit(autoDestroy);               /*to call destroy() before exit*/
    }
    else {
        log_info("Platform has been initialized");
    }
}

device_param_t Plat::get_device_param() {
    if (Plat::_instance == nullptr) {
        log_error("Platform and Deivce have not been initialized");
        exit(1);
    }
    else {
        uint idx = Plat::_instance->_chosen_device_id;
        return Plat::_instance->_device_params;
    }
}

void Plat::init_properties() {
    log_info("------ Start hardware checking ------");
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
    status = clGetPlatformIDs(MAX_PLATFORM_NUM, &this->_platform, nullptr);
    memset(platform_name, '\0', sizeof(char)*200);
    status = clGetPlatformInfo(this->_platform, CL_PLATFORM_NAME, 200, platform_name, nullptr);
    log_info("Platform: %s", platform_name);
    this->_device_params.platform = this->_platform;

    /*get device IDs*/
    status = clGetDeviceIDs(this->_platform, this->_type, 0, 0, &this->_num_devices);
    checkErr(status, "No devices available");
    status = clGetDeviceIDs(this->_platform, this->_type, this->_num_devices, devices, nullptr);
    log_info("Number of devices: %d", this->_num_devices);

    for(int i = 0; i< this->_num_devices; i++) {
        memset(devices_name[i], '\0', sizeof(char)*200);
        status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 200, devices_name[i], nullptr);
        log_info("\tComputing device %d : %s", i, devices_name[i]);
    }

    log_info("Please enter the index of the device to use (0,1,2...) : ");

    cin >> this->_chosen_device_id;
//    this->chosen_device_id = 0;
    if (this->_chosen_device_id < 0 || this->_chosen_device_id >= _num_devices)   {
        log_error("Wrong parameter.");
        exit(1);
    }
    cl_device_id my_device = devices[this->_chosen_device_id];
    log_info("Selected device: %s", devices_name[this->_chosen_device_id]);
    this->_device_params.device = my_device;

    clGetDeviceInfo(my_device, CL_DEVICE_VERSION, sizeof(char)*100, cl_version_info, nullptr); /*retrieve the OpenCL support version*/

    /*create the context*/
    const cl_context_properties prop[3] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(this->_platform),0};
    cl_context my_context = clCreateContext(prop, 1, &my_device, nullptr, nullptr, &status);
    checkErr(status, "Fail to create the context."); //only 1 device is used
    this->_device_params.context = my_context;

    /*create the command queue*/
    cl_command_queue my_queue = clCreateCommandQueue(my_context, my_device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkErr(status, "Failed to create the command queue.");
    this->_device_params.queue = my_queue;

    /*initialize other params*/
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &this->_device_params.gmem_size, nullptr);  /*global memory size*/
    clGetDeviceInfo(my_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(uint64_t), &this->_device_params.cacheline_size, nullptr); /*cacheline size*/
    clGetDeviceInfo(my_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &this->_device_params.lmem_size, nullptr); /*local memory size*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint64_t), &this->_device_params.cus, nullptr);       /*number of CUs*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(uint64_t), &this->_device_params.max_alloc_size, nullptr);       /*number of CUs*/
    clGetDeviceInfo(my_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &this->_device_params.max_local_size, nullptr);       /*maximal local size*/

    /*get the wavefront size according to the device type*/
    cl_device_type my_type;
    clGetDeviceInfo(my_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &my_type, nullptr);

    /*a simple kernel*/
    cl_kernel temp_kernel = get_kernel(this->_device_params.device, this->_device_params.context, "mem_kernel.cl", "scale_mixed");

    if (my_type == CL_DEVICE_TYPE_GPU) {    /*GPUs*/
        clGetKernelWorkGroupInfo(temp_kernel, this->_device_params.device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(uint64_t), &this->_device_params.wavefront, nullptr);
    }
    else {      /*CPUs and MICs*/
//        clGetDeviceInfo(this->device_params.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(uint64_t), &this->device_params.wavefront, nullptr);
        clGetKernelWorkGroupInfo(temp_kernel, this->_device_params.device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(uint64_t), &this->_device_params.wavefront, nullptr);

    }

    /*display the params*/
    log_info("Version: %s", cl_version_info);
    log_info("Global memory size: %.1f GB", this->_device_params.gmem_size*1.0/1024/1024/1024);
    log_info("Local memory size: %.1f KB", this->_device_params.lmem_size*1.0/1024);
    log_info("Compute units: %d", this->_device_params.cus);
    log_info("Maximal local_size: %d", this->_device_params.max_local_size);
    log_info("Maximal memory object size: %.1f GB", this->_device_params.max_alloc_size*1.0/1024/1024/1024);
    log_info("Global memory cache line size: %d bytes", _device_params.cacheline_size);
    log_info("Wavefront size: %d", _device_params.wavefront);

    log_info("------ End of hardware checking ------");
}