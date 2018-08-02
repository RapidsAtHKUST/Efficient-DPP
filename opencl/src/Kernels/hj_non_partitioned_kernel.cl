#ifndef HJ_KERNEL_CL
#define HJ_KERNEL_CL

#include "params.h"

//non-partitioned hash join
//build a shared hash table
//number of buckets: at least twice the cardinality of the build table
kernel void build ( global const int* d_R_keys,
                    global const int* d_R_values,
                    const int r_len,
                    global int *d_table_keys,
                    global int *d_table_values,
                    const int hash_bits)        //number of bits used for hashing
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    int idx;
    unsigned table_len = 1<<hash_bits;
    unsigned mask = table_len-1;

    //initialize the hash table keys
    idx = globalId;
    while (idx < table_len) {
        d_table_keys[idx] = -1;     //initialize to -1
        idx += globalSize;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    idx = globalId;
    while (idx < r_len) {       //loop once and build the hash table
        int d_key = d_R_keys[idx];
        int d_value = d_R_values[idx];
        int hashed_key = d_key & mask;      //hash function

        while (atomic_cmpxchg(d_table_keys+hashed_key ,-1,d_key) != -1) {
            hashed_key = (hashed_key + 1) & mask;
        }
        d_table_values[hashed_key] = d_value;
        idx += globalSize;
    }
}

kernel void probe(global const int* d_S_keys,
                  global const int* d_S_values,
                  const int s_len,
                  global int *d_table_keys,
                  global int *d_table_values,
                  const int hash_bits,
                  global int *d_out)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    int idx;
    unsigned table_len = 1<<hash_bits;
    unsigned mask = table_len-1;

    int res = 0;

    idx = globalId;
    while (idx < s_len) {       //loop once and probe in the hash table
        int d_key = d_S_keys[idx];
        int d_value = d_S_values[idx];
        int hashed_key = d_key & mask;      //hash function

        int cur_key = d_table_keys[hashed_key];
        while (cur_key != -1) {
            if (cur_key == d_key)
                res++;
            hashed_key = (hashed_key + 1) & mask;
            cur_key = d_table_keys[hashed_key];
        }
        idx += globalSize;
    }
    atomic_add(&d_out[0], res);
}

#endif