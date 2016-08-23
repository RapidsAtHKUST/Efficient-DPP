#ifndef BARRIER_KERNEL_CL
#define BARRIER_KERNEL_CL

kernel void barrier_free (
    global float* d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);
    const float scalar1 = 1.01, scalar2 = 0.001;

    float temp = d_values[globalId];
    for(int i = 0; i < repeatTime; i++) {
        temp = temp * scalar1 + scalar2;
    }
    d_values[globalId] = temp;
}

kernel void barrier_in (
    global float* d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);
    const float scalar1 = 1.01, scalar2 = 0.001;

    float temp = d_values[globalId];
    for(int i = 0; i < repeatTime; i++) {
        temp = temp * scalar1 + scalar2;
    	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    d_values[globalId] = temp;
}

#endif