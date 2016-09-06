#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//mem_read: repeat 20 times, unrolled
#define READ_REPEAT         (20)
#define STRIDED_ADD(time)   (v += d_source_values[begin + time]);
#define COALESCED_ADD       v += d_source_values[globalId];     \
                            globalId += globalSize;


kernel void mem_read (
    global const TYPE* restrict d_source_values,
    global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    TYPE v = 0.0;

    //strided unrolling for Xeon & Xeon Phi
    // int begin = globalId * READ_REPEAT;
    // STRIDED_ADD(0); STRIDED_ADD(1); STRIDED_ADD(2);
    // STRIDED_ADD(3); STRIDED_ADD(4); STRIDED_ADD(5);
    // STRIDED_ADD(6); STRIDED_ADD(7); STRIDED_ADD(8);
    // STRIDED_ADD(9); STRIDED_ADD(10); STRIDED_ADD(11);
    // STRIDED_ADD(12); STRIDED_ADD(13); STRIDED_ADD(14);
    // STRIDED_ADD(15); STRIDED_ADD(16); STRIDED_ADD(17);
    // STRIDED_ADD(18); STRIDED_ADD(19); 

    //coalesced unrolling for GPU
    int globalSize = get_global_size(0);
    int global_output = globalId;
    COALESCED_ADD; COALESCED_ADD; COALESCED_ADD; COALESCED_ADD;
    COALESCED_ADD; COALESCED_ADD; COALESCED_ADD; COALESCED_ADD;
    COALESCED_ADD; COALESCED_ADD; COALESCED_ADD; COALESCED_ADD;
    COALESCED_ADD; COALESCED_ADD; COALESCED_ADD; COALESCED_ADD;
    COALESCED_ADD; COALESCED_ADD; COALESCED_ADD; COALESCED_ADD;
    // for(int i = 0; i < 20; i++) {
    //     v += d_source_values[globalId];
    //     globalId += globalSize;
    // }
    // COALESCED_ADD;

    d_dest_values[global_output] = v;
}

kernel void mem_write (global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = 123.45f;
}

kernel void mem_mul (
    global const TYPE* restrict d_source_values, 
    global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId] * 0.3f;
}

kernel void mem_triad(
    global const TYPE * restrict d_source_values1,
    global const TYPE * restrict d_source_values2,
    global TYPE * restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values1[globalId] + 0.3f * d_source_values2[globalId];
}

kernel void mem_mul_coalesced (
    global const TYPE* restrict d_source_values, 
    global TYPE* restrict d_dest_values,
    const int repeat)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for(int i = 0; i < repeat; i++) {
        d_dest_values[globalId] = d_source_values[globalId] * 0.3f;
        globalId += globalSize;
    }
}

kernel void mem_mul_strided (
    global const TYPE* restrict d_source_values, 
    global TYPE* restrict d_dest_values,
    const int repeat)
{
    int globalId = get_global_id(0);

    int begin = globalId * repeat;
    for(int i = 0; i < repeat; i++) {
        d_dest_values[begin+i] = d_source_values[begin+i] * 0.3f ;
    }
}

#endif