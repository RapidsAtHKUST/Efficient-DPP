#include "general.h"
#include "log.h"
using namespace std;

/*OpenCL related functions*/
void checkErr(cl_int status, const char* name, int tag) {
    if (status != CL_SUCCESS) {
        log_error("StatusError: %s (%d) Tag: %d", name, status, tag);
        exit(EXIT_FAILURE);
    }
}

void cl_mem_free(cl_mem object) {
    if (object != 0 || object != NULL) {
        cl_int status = clReleaseMemObject(object);
        checkErr(status, "Failed to release the device memory object.");
    }
}

double clEventTime(const cl_event event){
    cl_ulong start,end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
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
    cout<<"compilation log:"<<endl;
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    char *log = new char[log_size];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    cout<<log<<endl;

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
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)(&source), 0, &status);
    checkErr(status, "Failed to creat program.");

    char *args = new char[1000];
    memset(args, '\0', sizeof(char)*1000);
    strcat(args, "-I");
    strcat(args, PROJECT_ROOT);
    strcat(args, "/inc ");

    strcat(args, "-I");
    strcat(args, PROJECT_ROOT);
    strcat(args, "/src/kernels ");

    strcat(args," -DKERNEL ");

    if (params != NULL) strcat(args, params);

//    strcat(args, " -auto-prefetch-level=0 ");
    status = clBuildProgram(program, 1, &device, args, 0, 0);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        cerr<<"\tCompilation error."<<endl;
        display_compilation_log(device, program);
        exit(EXIT_FAILURE);
    }

//extract the assembly programs if necessary
    size_t ass_size;
    status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &ass_size, NULL);
    checkErr(status,"Failed to get the size of the assembly program.");

    unsigned char *binary = new unsigned char[ass_size];
    status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, ass_size, &binary, NULL);
    checkErr(status,"Failed to generate the assembly program.");

    FILE * fpbin = fopen( "assembly.ass", "w" );
    if( fpbin == NULL )
    {
        fprintf( stderr, "Cannot create '%s'\n", "assembly.ass" );
    }
    else
    {
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