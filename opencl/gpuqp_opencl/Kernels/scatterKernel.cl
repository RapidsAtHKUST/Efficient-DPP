#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

#include "dataDefinition.h"

kernel void scatterKernel(global const Record* source,
                          global Record* dest,
                          const uint length,
                          global const uint* loc)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    for(int pos = globalId; pos < length; pos += globalSize) {
        dest[loc[pos]] = source[pos];
    }
}

#endif