#include "general.h"

/*OpenCL related functions*/
void checkErr(cl_int status, const char* name, int tag) {
    if (status != CL_SUCCESS) {
        std::cout<<"statusError: " << name<< " (" << status <<") Tag: "<<tag<<std::endl;
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