#define BITS 8              //sort 8 bits in a pass
#define RADIX (1<<BITS)
#include "dataDefinition.h"

kernel void countHis(const global Record* source,
                     const uint length,
                     global uint* histogram,        //size: globalSize * RADIX
                     local ushort* temp,           //each group has temp size of BLOCKSIZE * RADIX
                     const uint shiftBits)
{
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int elePerThread = ceil(1.0*length / globalSize);
    int offset = localId * RADIX;
    uint mask = RADIX - 1;
    
    //initialization
    for(int i = 0; i < RADIX; i++) {
        temp[i + offset] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[id].y;
        current = (current >> shiftBits) & mask;
        temp[offset + current]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int i = 0; i < RADIX; i++) {
        histogram[i*globalSize + globalId] = (uint)temp[offset+i];
    }
}

kernel void writeHis(const global Record* source,
                     const uint length,
                     const global uint* histogram,
                     global uint* loc,              //size equal to the size of source
                     local uint* temp,
                     const uint shiftBits)               //each group has temp size of BLOCKSIZE * RADIX
{
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int elePerThread = ceil(1.0 *length / globalSize);     // length for each thread to proceed
    int offset = localId * RADIX;
    uint mask = RADIX - 1;
    
    for(int i = 0; i < RADIX; i++) {
        temp[offset + i] = histogram[i*globalSize + globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[globalId * elePerThread + i].y;
        current = (current >> shiftBits) & mask;
        loc[globalId * elePerThread + i] = temp[offset + current];
        temp[offset + current]++;
    }
}

