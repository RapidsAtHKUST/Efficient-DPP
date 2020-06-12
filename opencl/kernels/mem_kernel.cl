#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

#include "../params.h"

#define SCALAR                  (3)

/* Memory bandwidth and access patterns test */
kernel
void copy_bandwidth(global const int* d_in, global int* d_out) {
    int globalId = get_global_id(0);
    d_out[globalId] = d_in[globalId];
}

kernel
void addition_bandwidth(global const int* d_in_1, global const int *d_in_2, global int* d_out) {
    int globalId = get_global_id(0);
    d_out[globalId] = d_in_1[globalId] + d_in_2[globalId];
}

kernel
void scale_bandwidth (global const int* d_in, global int* d_out) {
    int globalId = get_global_id(0);
    d_out[globalId] = SCALAR * d_in[globalId];
}

kernel
void triad_bandwidth(global const int* d_in_1, global const int *d_in_2, global int* d_out) {
    int globalId = get_global_id(0);
    d_out[globalId] = d_in_1[globalId] + SCALAR * d_in_2[globalId];
}

kernel
void scale_column (global const int* d_in,
                   global int* d_out,
                   const int repeat) {
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    for(int i = 0; i < repeat; i++) {
        d_out[globalId] = d_in[globalId] * SCALAR;
        globalId += globalSize;
    }
}

kernel
void scale_row (global const int* d_in,
                global int* d_out,
                const int repeat) {
    int globalId = get_global_id(0);
    int begin = globalId * repeat;
    for(int i = 0; i < repeat; i++) {
        d_out[begin+i] = d_in[begin+i] * SCALAR ;
    }
}

kernel
void scale_mixed(global const int* d_in,
                  global int* d_out,
                  const int repeat) {
    int globalId = get_global_id(0);
    int warp_num = globalId >> WARP_BITS;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));
    for(int i = 0; i < repeat; i++) {
        d_out[idx] = d_in[idx] * SCALAR;
        idx += WARP_SIZE;
    }
}

kernel
void wg_access( global int *in,
                const int len,              //len for each wg
                global int *atom,           //for atomic updates
                global int *idx_arr,        //filled with wg index orders
                global int *out) {          //a single output slot
    int w, w_idx, begin, end;
    long acc = 0;
    w_idx = atomic_inc(atom);
    w = idx_arr[w_idx];
    begin = w * len;
    end = (w+1) * len;
    for(int c = begin; c < end; c++) {
        acc += (in[c] & 0b1);
    }
    out[0] = (int)acc; //to avoid optimizing away the add operations
}

kernel
void cache_heat(global int *data, int length) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int global_warp_id = global_id >> WARP_BITS;
    int lane = local_id & MASK;

    int ele_per_wi = (length + global_size - 1) / global_size;

    int begin = global_warp_id * WARP_SIZE * ele_per_wi;
    int end = (global_warp_id + 1) * WARP_SIZE * ele_per_wi;
    if (end >= length) end = length;

    int c = begin + lane;
    while (c < end) {
        data[c] = c;     //write only
        c += WARP_SIZE;
    }
}
#endif