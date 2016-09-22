#ifndef GATHER_KERNEL_CL
#define GATHER_KERNEL_CL

kernel void gatherKernel( 
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
  global const int *d_source_values, 
  global int* d_dest_values,
  const int length_input, const int length_output, global const int* loc,
  int ele_per_thread)
{
    int globalId = get_global_id(0);

    int warpSize = 16;
    int warpId = globalId >> 4;

    int begin = warpId * warpSize * ele_per_thread + (globalId & (warpSize-1));
    int end = ((warpId + 1) * warpSize * ele_per_thread < length_output)? ((warpId + 1) * warpSize * ele_per_thread) : length_output;

    int numOfRun = 1;
    int numPerRun = length_input / numOfRun;
    if (length_input % numOfRun != 0) numPerRun += 1;

    for(int i = 0; i < numOfRun; i++) {

      int from = i * numPerRun;
      int to = (i+1) * numPerRun;

      for(int pos = begin; pos < end; pos += warpSize) {
          int tempLoc = loc[pos];
          if (tempLoc >= from && tempLoc < to) {
            d_dest_values[pos] = d_source_values[tempLoc];
            #ifdef RECORDS
              if (isRecord)
                d_dest_keys[pos] = d_source_keys[tempLoc];
            #endif
          }
      }
    }
}

#endif
