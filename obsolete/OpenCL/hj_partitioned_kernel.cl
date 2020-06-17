#ifndef HJ_KERNEL_CL
#define HJ_KERNEL_CL

#include "params.h"

// # of work-groups is equal to the number of partitions (buckets) in R and S
//the start_R and start_S array are generated through 1024 work-groups
kernel void build_probe (   global const int* d_R_keys,
                            global const int* d_R_values,
                            const int r_len,
                            global const int* d_S_keys,
                            global const int* d_S_values,
                            const int s_len,
                            global const int *start_R,
                            global const int *start_S,
                            global int *d_out,
                            local int* d_R_local_keys,        //size: 15.5KB
                            local int* d_R_local_values,
                            local int* d_S_local_keys,
                            local int* d_S_local_values)        //size: 15.5KB
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned r_begin, r_end, s_begin, s_end;

    r_begin = start_R[groupId];
    s_begin = start_S[groupId];

    if (groupId == groupNum - 1) {
        r_end = r_len;
        s_end = s_len;
    }
    else {
        r_end = start_R[groupId+1];
        s_end = start_S[groupId+1];
    }

    unsigned r_par_len = r_end - r_begin;
    unsigned s_par_len = s_end - s_begin;

//1.copy d_R partitions to the local memory
    int i = r_begin + localId;
    while (i < r_end) {
        d_R_local_keys[i-r_begin] = d_R_keys[i];
        d_R_local_values[i-r_begin] = d_R_values[i];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

//2.build the hash table from S
    //initialize the hash table
    int hash_total = 15.5*1024/sizeof(int);
    i = localId;
    while (i < hash_total) {
        d_S_local_keys[i] = -1;        //default key (empty)
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //s partition is at most 16KB(2048 tuples)
    //open addressing radix hashing using the middle 11 bits (exclude the right most 13 bits)
    int collision = 0;
    i = s_begin + localId;
    while (i < s_end) {
        int myKey = ((d_S_keys[i] >> 13) & 0b11111111111)<<1;   //radix hashing
        int d_S_key = d_S_keys[i];
        int delta = 1;
        while (atomic_cmpxchg(d_S_local_keys+myKey ,-1,d_S_key) != -1) {
            collision ++;
            myKey = (myKey+delta*delta)%hash_total;
            delta ++;
        }
        d_S_local_values[myKey] = d_S_keys[i];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

//3.probe
    //R as the outer relation
    int res = 0;
    i = localId;
    while (i < r_par_len) {
        int d_R_local_key = d_R_local_keys[i];
        int myKey = ((d_R_local_key >> 13) & 0b11111111111)<<1;
        int delta = 1;

        int d_S_local_key = d_S_local_keys[myKey];                  //gather
        while (d_S_local_key != -1) {
            if (d_R_local_key == d_S_local_key)
                res++;
            myKey = (myKey + delta*delta)%hash_total;
            d_S_local_key = d_S_local_keys[myKey];
            delta++;
        }
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (groupId == 1)
    atomic_add(&d_out[0], res);
}


#endif