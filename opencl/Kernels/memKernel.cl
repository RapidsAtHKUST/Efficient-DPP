#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//warp_bits: 5 for GPU, 4 for Xeon Phi and 3 for Xeon CPU
#define WARP_BITS               (3)
#define WARP_SIZE               (1<<WARP_BITS)
#define MASK                    (WARP_SIZE-1)
#define SCALAR                  (3)

//testing memory bandwidth and access patterns on different devices
kernel void mul_bandwidth (global const int* d_in, global int* d_out, const int scalar)
{
    int globalId = get_global_id(0);
    d_out[globalId] = scalar * d_in[globalId];
}

kernel void mul_column_based (
    global const int* d_in,
    global int* d_out,
    const int repeat)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for(int i = 0; i < repeat; i++) {
        d_out[globalId] = d_in[globalId] * SCALAR;
        globalId += globalSize;
    }
}

kernel void mul_row_based (
    global const int* d_in,
    global int* d_out,
    const int repeat)
{
    int globalId = get_global_id(0);

    int begin = globalId * repeat;
    for(int i = 0; i < repeat; i++) {
        d_out[begin+i] = d_in[begin+i] * SCALAR ;
    }
}

//attention: should set the WARP_SIZE before testing on a device!!
kernel void mul_mixed (
    global const int* d_in,
    global int* d_out,
    const int repeat)
{
    //for Nvidia GPU, warpsize = 32, for Xeon Phi, warpsize = 16, for
    int globalId = get_global_id(0);
    int warp_num = globalId >> WARP_BITS;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        d_out[idx] = d_in[idx] * SCALAR;
        idx += WARP_SIZE;
    }
}

kernel void wg_access(
    global int *data,
    const int length,
    global int *atom,       //for atomic updates
    global int *idx_arr)     //filled with wg index orders
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int local_warp_id = local_id >> WARP_BITS;
    int lane = local_id & MASK;
    int ele_per_wi = (length + global_size - 1) / global_size;

    local int w;
    local int w_idx;
    if (local_id == 0)  {
        w_idx = atomic_inc(atom);
        w = idx_arr[w_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int global_begin = local_size * ele_per_wi * w;
    int begin = local_warp_id * WARP_SIZE * ele_per_wi;
    int end = (local_warp_id + 1) * WARP_SIZE * ele_per_wi;
    if (global_begin + end >= length) end = length - global_begin;

    //for read-only
    long acc = 0;
    int c = begin + lane;
    while (c < end) {
        acc += (data[global_begin+c] & 0b1);
        c += WARP_SIZE;
    }
    if(global_id == 0)  data[0] = acc;
}

kernel void wg_access_no_atomic(
    global int *data,
    const int length,
    int num_of_groups)     //filled with wg index orders
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int local_warp_id = local_id >> WARP_BITS;
    int lane = local_id & MASK;
    int ele_per_wi = (length + num_of_groups - 1) / num_of_groups / local_size;

    int group_id = get_group_id(0);
    int group_size = get_num_groups(0);
    for(int w = group_id; w < num_of_groups; w += group_size) {
        int global_begin = local_size * ele_per_wi * w;
        int begin = local_warp_id * WARP_SIZE * ele_per_wi;
        int end = (local_warp_id + 1) * WARP_SIZE * ele_per_wi;
        if (global_begin + end >= length) end = length - global_begin;

        //for read-only
        long acc = 0;
        int c = begin + lane;
        while (c < end) {
            acc += (data[global_begin+c] & 0b1);
            c += WARP_SIZE;
        }
        if(global_id == 0)  data[0] = acc;
    }
}



kernel void cache_heat(
        global int *data,
        int length)
{
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