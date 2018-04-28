#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

#include "DataDef.h"

kernel void scatterKernel(global const int *d_in,
                         global int* d_out,
                         global const int* loc,
                         const int length,
                         const int ele_per_thread,
                         const int from,        //start position
                         const int to)          //end position)
{
    int globalId = get_global_id(0);
    int warpId = globalId >> WARP_BITS;

    int begin = warpId * WARP_SIZE * ele_per_thread + (globalId & (WARP_SIZE-1));
    int end = ((warpId + 1) * WARP_SIZE * ele_per_thread < length)? ((warpId + 1) * WARP_SIZE * ele_per_thread) : length;

    for(int i = begin; i < end; i += WARP_SIZE) {
        int pos = loc[i];
        if (pos >= from && pos < to) {
            d_out[pos] = d_in[i];
        }
    }
}

#endif