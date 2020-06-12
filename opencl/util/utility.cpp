//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#include "../primitives.h"
#include "log.h"
#include <omp.h>
using namespace std;

double diffTime(struct timeval end, struct timeval start) {
	return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

void my_itoa(int num, char *buffer, int base) {
    int len=0, p=num;
    while(p/=base) len++;
    len++;
    for(p=0;p<len;p++)
    {
        int x=1;
        for(int t=p+1;t<len;t++) x*=base;
        buffer[p] = num/x +'0';
        num -=(buffer[p]-'0')*x;
    }
    buffer[len] = '\0';
}

/*calculating the memory bandwidth
 *elasped time: in ms using diffTime*/
double compute_bandwidth(uint64_t num, int wordSize, double kernel_time) {
    return 1.0*num/1024/1024/1024*wordSize/kernel_time * 1000 ;
}

bool pair_cmp (pair<double, double> i , pair<double, double> j) {
    return i.first < j.first;
}

double average_Hampel(double *input, int num) {
    int valid = 0;
    double total = 0;

    double *temp_input = new double[num];
    vector< pair<double, double> > myabs_and_input_list;

    double mean, abs_mean;

    for(int i = 0; i < num; i++) temp_input[i]=input[i];

    sort(temp_input, temp_input+num);
    if (num % 2 == 0)  mean = 0.5*(temp_input[num/2-1] + temp_input[num/2]);
    else               mean = temp_input[(num-1)/2];

    for(int i = 0; i < num; i++)
        myabs_and_input_list.push_back(make_pair(fabs(temp_input[i]-mean),temp_input[i]));

    typedef vector< pair<double, double> >::iterator VectorIterator;
    sort(myabs_and_input_list.begin(), myabs_and_input_list.end(), pair_cmp);

    if (num % 2 == 0)  abs_mean = 0.5*(myabs_and_input_list[num/2-1].first + myabs_and_input_list[num/2].first);
    else               abs_mean = myabs_and_input_list[(num-1)/2].first;

    abs_mean /= 0.6745;

    for(VectorIterator iter = myabs_and_input_list.begin(); iter != myabs_and_input_list.end(); iter++) {
        if (abs_mean == 0) { /*if abs_mean=0,only choose those with abs=0*/
            if (iter->first == 0) {
                total += iter->second;
                valid ++;
            }
        }
        else {
            double div = iter->first / abs_mean;
            if (div <= 3.5) {
                total += iter->second;
                valid ++;
            }
        }
    }
    total = 1.0 * total / valid;
    if(temp_input)  delete[] temp_input;
    return total;
}

/*OpenCL related functions*/
void checkErr(cl_int status, const char* name, int tag) {
    if (status != CL_SUCCESS) {
        log_error("StatusError: %s (%d) Tag: %d", name, status, tag);
        exit(EXIT_FAILURE);
    }
}

void cl_mem_free(cl_mem object) {
    if (object != 0 || object != nullptr) {
        cl_int status = clReleaseMemObject(object);
        checkErr(status, "Failed to release the device memory object.");
    }
}

double clEventTime(const cl_event event){
    cl_ulong start,end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    return (end - start) / 1000000.0;
}

void add_param(char *param, char *macro, bool has_value, int value) {
    strcat(param, " -D");
    strcat(param, macro);

    if (has_value) {
        char value_str[100]={'\0'};
        my_itoa(value, value_str, 10);  /*translated to string*/
        strcat(param, "=");
        strcat(param, value_str);
    }
    strcat(param, " ");
}

void display_compilation_log(cl_device_id device, cl_program program) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    char *log = new char[log_size];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
    log_info("compilation log: %s", log);
    if (log)    delete[] log;
}

cl_kernel get_kernel(
        cl_device_id device, cl_context context,
        char *file_name, char *func_name, char *params) {

/*read the raw kernel file*/
    char *addr = new char[1000];
    memset(addr, '\0', sizeof(char)*1000);
    strcat(addr, PROJECT_ROOT);
    strcat(addr, "/kernels/");
    strcat(addr, file_name);

    ifstream in(addr,std::fstream::in| std::fstream::binary);
    if(!in.good()) {
        log_error("Kernel file not exist");
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
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)(&source), 0, &status);
    checkErr(status, "Failed to creat program.");

    char *args = new char[1000];
    memset(args, '\0', sizeof(char)*1000);
    strcat(args, "-I");
    strcat(args, PROJECT_ROOT);
    strcat(args, "/kernels ");
    strcat(args," -DKERNEL ");

    if (params != nullptr) strcat(args, params);

//    strcat(args, " -auto-prefetch-level=0 ");
    status = clBuildProgram(program, 1, &device, args, 0, 0);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        cerr<<"\tCompilation error."<<endl;
        display_compilation_log(device, program);
        exit(EXIT_FAILURE);
    }

    /*extract the assembly programs if necessary*/
    size_t ass_size;
    status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &ass_size, nullptr);
    checkErr(status,"Failed to get the size of the assembly program.");

    unsigned char *binary = new unsigned char[ass_size];
    status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, ass_size, &binary, nullptr);
    checkErr(status,"Failed to generate the assembly program.");

    FILE * fpbin = fopen( "assembly.ass", "w" );
    if( fpbin == nullptr ) {
        fprintf( stderr, "Cannot create '%s'\n", "assembly.ass" );
    }
    else {
        fwrite( binary, 1, ass_size, fpbin );
        fclose( fpbin );
    }
    delete [] binary;


    /*create the kernel*/
    cl_kernel kernel = clCreateKernel(program, func_name, &status);
    checkErr(status, "Kernel function name not found.");

    if(source)  delete[] source;
    if(addr)    delete[] addr;
    if(args)    delete[] args;

    in.close();
    return kernel;
}

/*data generators*/
/*
 * Generate random uniform int value array
 * */
void random_generator_int(int *keys, uint64_t length, int max, unsigned long long seed) {
#pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int my_seed = seed + tid;
#pragma omp for schedule(dynamic)
        for(int i = 0; i < length ; i++)    keys[i] = rand_r(&my_seed) % max;
    }
}

/*
 * Generate random uniform unique int value array
 * */
void random_generator_int_unique(int *keys, uint64_t length) {
    srand((unsigned)time(nullptr));
#pragma omp parallel for
    for(int i = 0; i < length ; i++) {
        keys[i] = i;
    }
    log_trace("Key assignment finished");
    /*shuffling*/
    for(auto i = length-1; i > 0; i--) {
        auto from = rand() % i;
        auto to = rand() % i;
        std::swap(keys[from], keys[to]);
    }
    log_trace("Key shuffling finished");
}