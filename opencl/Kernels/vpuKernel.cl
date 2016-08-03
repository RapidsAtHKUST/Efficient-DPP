#ifndef VPU_KERNEL_CL
#define VPU_KERNEL_CL

#define MADD1_OP  temp = temp * con + con;

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP

kernel void vpu1 (
    global float* d_values,
    int length,
    int con, 
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/1) {
        float temp = d_values[globalId];
        for(int i = 0; i < repeatTime; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_values[globalId] = temp;
        globalId += globalSize;
    }
}

kernel void vpu2 (
    global float2* d_values,
    int length,
    int con, 
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/2) {
        float2 temp = d_values[globalId];
        for(int i = 0; i < repeatTime; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_values[globalId] = temp;
        globalId += globalSize;
    }
}

kernel void vpu4 (
    global float4* d_values,
    int length,
    int con, 
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/4) {
        float4 temp = d_values[globalId];
        for(int i = 0; i < repeatTime; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_values[globalId] = temp;
        globalId += globalSize;
    }
}

kernel void vpu8 (
    global float8* d_values,
    int length,
    int con, 
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/8) {
        float8 temp = d_values[globalId];
        for(int i = 0; i < repeatTime; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_values[globalId] = temp;
        globalId += globalSize;
    }
}

kernel void vpu16 (
    global float16* d_values,
    int length,
    int con, 
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/16) {
        float16 temp = d_values[globalId];
        for(int i = 0; i < repeatTime; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_values[globalId] = temp;
        globalId += globalSize;
    }
}

#endif