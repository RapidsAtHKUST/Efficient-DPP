#ifndef SCATTER_KERNEL_CL
#define SCATTER_KERNEL_CL

kernel void scatterKernel( 
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
	global const int16 *d_source_values, 
	global int* d_dest_values,
    const int length, global const int16 * loc,
    int ele_per_thread)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int max = length;
    if ((globalId + 1) * ele_per_thread < max)  max = (globalId + 1) * ele_per_thread;
    for(int pos = globalId * ele_per_thread;  pos < max; pos += 16) {
        
        //int targetLoc = loc[pos];
        int value_index = pos / 16;
        d_dest_values[loc[value_index].s0] = d_source_values[value_index].s0;
        d_dest_values[loc[value_index].s1] = d_source_values[value_index].s1;
        d_dest_values[loc[value_index].s2] = d_source_values[value_index].s2;
        d_dest_values[loc[value_index].s3] = d_source_values[value_index].s3;
        d_dest_values[loc[value_index].s4] = d_source_values[value_index].s4;
        d_dest_values[loc[value_index].s5] = d_source_values[value_index].s5;
        d_dest_values[loc[value_index].s6] = d_source_values[value_index].s6;
        d_dest_values[loc[value_index].s7] = d_source_values[value_index].s7;

        d_dest_values[loc[value_index].s8] = d_source_values[value_index].s8;
        d_dest_values[loc[value_index].s9] = d_source_values[value_index].s9;
        d_dest_values[loc[value_index].sa] = d_source_values[value_index].sa;
        d_dest_values[loc[value_index].sb] = d_source_values[value_index].sb;
        d_dest_values[loc[value_index].sc] = d_source_values[value_index].sc;
        d_dest_values[loc[value_index].sd] = d_source_values[value_index].sd;
        d_dest_values[loc[value_index].se] = d_source_values[value_index].se;
        d_dest_values[loc[value_index].sf] = d_source_values[value_index].sf;
    #ifdef RECORDS
    	if (isRecord)
        	d_dest_keys[targetLoc] = d_source_keys[pos];
    #endif
    }
}

#endif