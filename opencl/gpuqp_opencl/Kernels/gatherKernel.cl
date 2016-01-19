#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

#include "dataDefinition.h"

kernel void gatherKernel( global const Record* reSource,
                          global Record* reDest,
                          const int length,
                          global const int* loc)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    for(int pos = globalId; pos < length; pos += globalSize) {
        int targetLoc = loc[pos];
        reDest[pos] = reSource[targetLoc];
    }
}

#endif