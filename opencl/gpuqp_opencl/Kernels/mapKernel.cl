#ifndef MAP_KERNEL_CL
#define MAP_KERNEL_CL

#include "dataDefinition.h"

int floorOfPower2(int a) {
    int base = 1;
    while (base < a) {
        base <<= 1 ;
    }
    return base >> 1;
}

//map with coalesced access
kernel void mapKernel ( global const Record* reSource,
                        const int length,
                        global Record* reDest)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length) {
        reDest[globalId].x = reSource[globalId].x;
        reDest[globalId].y = floorOfPower2(reSource[globalId].y);
        globalId += globalSize;
    }
}

#endif