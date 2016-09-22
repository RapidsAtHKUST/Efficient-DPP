#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

kernel void scatterKernel( 
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
  global const int *d_source_values, 
  global int* d_dest_values,
  const int length_input, const int length_output, global const int* loc,
  int ele_per_thread, const int numOfRun)
{
    int globalId = get_global_id(0);

    int warpSize = 32;
    int warpId = globalId >> 5;

    int begin = warpId * warpSize * ele_per_thread + (globalId & (warpSize-1));
    int end = ((warpId + 1) * warpSize * ele_per_thread < length_input)? ((warpId + 1) * warpSize * ele_per_thread) : length_input;

    int numPerRun = length_output / numOfRun;
    if (length_output % numOfRun != 0) numPerRun += 1;

    for(int i = 0; i < numOfRun; i++) {

      int from = i * numPerRun;
      int to = (i+1) * numPerRun;

      for(int pos = begin; pos < end; pos += warpSize) {
          int tempLoc = loc[pos];
          if (tempLoc >= from && tempLoc < to) {
            d_dest_values[tempLoc] = d_source_values[pos];
            #ifdef RECORDS
              if (isRecord)
                d_dest_keys[tempLoc] = d_source_keys[pos];
            #endif
          }
      }
    }
}

#endif