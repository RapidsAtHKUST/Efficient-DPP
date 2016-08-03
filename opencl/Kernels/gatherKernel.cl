#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

kernel void gatherKernel( 
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
	global const int *d_source_values, 
	global int* d_dest_values,
  const int length, global const int* loc,
  int ele_per_thread)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int max = length;
    if( (globalId+1) * ele_per_thread < length ) max = (globalId + 1) * ele_per_thread;
    for(int pos = globalId * ele_per_thread; pos < max; pos ++) {
        int targetLoc = loc[pos];
        d_dest_values[pos] = d_source_values[targetLoc];
    #ifdef RECORDS
    	if (isRecord)
        	d_dest_keys[pos] = d_source_keys[targetLoc];
    #endif
    }

}

#endif