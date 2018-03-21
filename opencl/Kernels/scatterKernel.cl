#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

#include "dataDef.h"

kernel void scatterKernel( 
  global const int *d_source_values, 
  global int* d_dest_values,
  global const int* loc,
  const int length, 
  int ele_per_thread, 
  const int numOfRun)
{
    int globalId = get_global_id(0);
    // int globalSize = get_global_size(0);

    int warpId = globalId >> WARP_BITS;

    int begin = warpId * WARP_SIZE * ele_per_thread + (globalId & (WARP_SIZE-1));
    int end = ((warpId + 1) * WARP_SIZE * ele_per_thread < length)? ((warpId + 1) * WARP_SIZE * ele_per_thread) : length;
    int x = 0;

    int numPerRun = (length + numOfRun - 1) / numOfRun;

    for(int i = 0; i < numOfRun; i++) {

      int from = i * numPerRun;
      int to = (i+1) * numPerRun;

      for(int pos = begin; pos < end; pos += WARP_SIZE) {
          int tempLoc = loc[pos];
          if (tempLoc >= from && tempLoc < to) {
             d_dest_values[tempLoc] = d_source_values[pos];
            // x = x + d_source_values[pos];
          }
      }
    }
    // d_dest_values[globalId] = x;
}

#endif