#ifndef VPU_KERNEL_CL
#define VPU_KERNEL_CL

//the TYPEing 
#define MADD1_OP  temp = 1 - temp *  ;

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP

kernel void vpu1 (
    global TYPE* restrict d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);

    TYPE temp = d_values[globalId] + (TYPE)(1);
    for(int i = 0; i < repeatTime; i++) {
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20
    }
    d_values[globalId] = temp;
}

kernel void vpu2 (
    global TYPE* restrict d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);

    TYPE2 temp = d_values[globalId] + (TYPE2)(1,2);
    for(int i = 0; i < repeatTime; i++) {
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20
    }
    d_values[globalId] = temp.s0 + temp.s1;
}

kernel void vpu4 (
    global TYPE* restrict d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);

    TYPE4 temp = d_values[globalId] + (TYPE4)(1,2,3,4);
    for(int i = 0; i < repeatTime; i++) {
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20
    }
    d_values[globalId] = temp.s0 + temp.s1 + temp.s2 + temp.s3;
}

kernel void vpu8 (
    global TYPE* restrict d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);

    TYPE8 temp = d_values[globalId] + (TYPE8)(1,2,3,4,5,6,7,8);
    for(int i = 0; i < repeatTime; i++) {
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20
    }
    d_values[globalId] = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7;
}

kernel void vpu16 (
    global TYPE* restrict d_values,
    int repeatTime)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    TYPE16 temp = d_values[globalId] + (TYPE16)(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
    for(int i = 0; i < repeatTime; i++) {
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20
    }
    d_values[globalId] = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 + temp.s6 + temp.s7 + temp.s8 + temp.s9 + temp.sa + temp.sb + temp.sc + temp.sd + temp.se + temp.sf;
}

#endif