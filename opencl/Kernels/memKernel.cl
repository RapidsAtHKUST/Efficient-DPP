#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//warp_bits: 5 for GPU, 4 for Xeon Phi and 3 for Xeon CPU
#define WARP_BITS            (3)
#define WARP_SIZE            (1<<WARP_BITS)

//testing memory bandwidth and access patterns on different devices
kernel void mem_mul_bandwidth (global const int* restrict d_in, global int* restrict d_out, const int scalar)
{
    int globalId = get_global_id(0);
    d_out[globalId] = scalar * d_in[globalId];
}

kernel void mem_mul_coalesced (
    global const int* restrict d_in,
    global int* restrict d_out,
    const int repeat)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for(int i = 0; i < repeat; i++) {
        d_out[globalId] = d_in[globalId] * 3;
        globalId += globalSize;
    }
}

kernel void mem_mul_strided (
    global const int* restrict d_in,
    global int* restrict d_out,
    const int repeat)
{
    int globalId = get_global_id(0);

    int begin = globalId * repeat;
    for(int i = 0; i < repeat; i++) {
        d_out[begin+i] = d_in[begin+i] * 3 ;
    }
}

//attention: should set the WARP_SIZE before testing on a device!!
kernel void mem_mul_strided_warpwise (
    global const int* restrict d_in,
    global int* restrict d_out,
    const int repeat)
{
    //for Nvidia GPU, warpsize = 32, for Xeon Phi, warpsize = 16, for
    int globalId = get_global_id(0);
    int warp_num = globalId >> WARP_BITS;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        d_out[idx] = d_in[idx] * 3 ;
        idx += WARP_SIZE;
    }
}

#endif