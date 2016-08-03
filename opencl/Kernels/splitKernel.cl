#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

#include "dataDefinition.h"

kernel void createList( global const Record* source,
                        int length,
                        global int *L,
                        local int* temp,        //size: fanout * localSize
                        int fanout)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for(int pos = 0; pos < fanout; pos ++) {
        temp[pos * localSize + localId] = 0;
    }
    
    for(int pos = globalId; pos < length; pos += globalSize) {
        int offset = source[pos].y;
        temp[offset * localSize + localId]++;
    }
    
    for(int pos = 0; pos < fanout; pos ++) {
        L[pos * globalSize + globalId] = temp[pos * localSize + localId];
    }
}

kernel void splitWithList(global const Record *source,
                          global int* L,
                          int length,
                          global Record *dest,
                          local int* temp,      //size: fanout * localSize
                          int fanout)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    for(int pos = 0; pos < fanout; pos ++) {
        temp[pos * localSize + localId] = L[pos* globalSize + globalId];
    }
    
    for(int pos = globalId; pos < length; pos += globalSize) {
        int offset = source[pos].y;
        dest[temp[offset * localSize + localId]++] = source[pos];
    }
}

//split for hash join, processing the first 12 digits
kernel void createListHJ( global const Record * source,
                          uint length,
                          global int * histogram,               //size: radix * blockNum * BLOCKSIZE
                          local int * temp,                     //size: raidx * BLOCKSIZE
                          uint bits,
                          uint shift)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int radix = 1 << bits;
    int mask = radix - 1;
    int elePerThread = ceil(1.0*length / globalSize);
    
    for(int pos = globalId; pos < radix * globalSize; pos += globalSize) {
        histogram[pos] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    for(int pos = localId; pos < radix * localSize; pos += localSize) {
        temp[pos] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = 0; pos < elePerThread; pos ++) {
        int id = globalId * elePerThread + pos;
        if (id >= length)   break;
        int cur = source[id].y;
        
        cur = ( cur >> shift ) & mask;
        temp[cur * localSize + localId] ++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int currentRadix = 0; currentRadix < radix; currentRadix++) {
        histogram[globalSize * currentRadix + globalId] = temp[localSize *currentRadix + localId];
    }
}

kernel void splitWithListHJ( global const Record * source,
                             global Record *dest,
                             uint length,
                             global int * histogram,                 //size: radix * blockNum * BLOCKSIZE
                             local int * temp,                       //size: raidx * BLOCKSIZE
                             uint bits,
                             uint shift)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    int radix = 1 << bits;
    int mask = radix - 1;
    int elePerThread = ceil(1.0*length / globalSize);
    
    for(int currentRadix = 0; currentRadix < radix; currentRadix++) {
        temp[currentRadix * localSize + localId] = histogram[currentRadix * globalSize + globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = 0; pos < elePerThread; pos ++) {
        int id = globalId * elePerThread + pos;
        if (id >= length)   break;
        int cur = source[id].y;

        cur = ( cur >> shift ) & mask;
        dest[temp[cur * localSize + localId]++] = source[id];
    }
}

#endif