#ifndef ATOMIC_KERNEL_CL
#define ATOMIC_KERNEL_CL

kernel void atomic_local (
    global int* restrict d_value,
    int repeatTime)
{
    int localId = get_local_id(0);
    int blockId = get_group_id(0);

    volatile local int tempSum;

    if (localId == 0)   tempSum = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < repeatTime; i++) {
        atomic_add(&tempSum, 1);
        // tempSum += 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0)  atomic_add(d_value, tempSum);
   // if (localId == 0)  d_value += tempSum;

}

kernel void atomic_global (
    global int* restrict d_value,
    int repeatTime)
{
    int globalId = get_global_id(0);

    for(int i = 0; i < repeatTime; i++) {
        atomic_add(d_value, 1);
        // d_value += 1;
    }
}

#endif

