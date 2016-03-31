#ifndef MAP_KERNEL_CL
#define MAP_KERNEL_CL

int floorOfPower2(int a) {
    int base = 1;
    while (base < a) {
        base <<= 1 ;
    }
    return base >> 1;
}

//map with coalesced access
kernel void mapKernel (
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
    global int* d_source_values,
    global int* d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length) {
        d_dest_values[globalId] = floorOfPower2(d_source_values[globalId]);
    #ifdef RECORDS
        if (isRecord)
            d_dest_keys[globalId] = d_source_keys[globalId];
    #endif
        globalId += globalSize;
    }
}

#endif