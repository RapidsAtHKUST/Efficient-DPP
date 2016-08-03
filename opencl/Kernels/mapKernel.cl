#ifndef MAP_KERNEL_CL
#define MAP_KERNEL_CL

#define MADD1_OP  s = s * a + a;

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP

int floorOfPower2_1(int a) {
    int base = 1;
    while (base < a) {
        base <<= 1 ;
    }
    return base >> 1;
}

int floorOfPower2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v>>1;
}

double calculatePi(int v) {
    double pi = 0;
    for(int k = 0; k <= 250; k++){
        //if(k % 2 == 0)  pi += 1.0/(2*k+1);
        //else            pi -= 1.0/(2*k+1); 

        pi += (k+v)/(2*v+1);
    }
    pi *= 4;
    return pi;
}

//map with coalesced access
kernel void mapKernel (
#ifdef RECORDS
   global int* d_source_keys,
   global int* d_dest_keys,
   bool isRecord,
#endif
    global float* d_source_values,
    global int* d_dest_values,
    int length,
    int k)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    float a = 1.01;

    while (globalId < length/1) {
        //d_dest_values[globalId] = (int)calculatePi(d_source_values[globalId]);
        //d_dest_values[globalId] = floorOfPower2(d_source_values[globalId]);
        
        //read-write
        // d_dest_values[globalId] = d_source_values[globalId] + 1;    
        
        //only write
        // d_dest_values[globalId] = globalId + 11;

        //only read

        float s = d_source_values[globalId];
        for(int i = 0; i < 60; i++) {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
            MADD1_MOP20 MADD1_MOP20
        }
        d_source_values[globalId] = s;

        //complicate calculate
        // float pi = 0;
        // int v = d_source_values[globalId];
        // for(int ki = 0; ki <= 0; ki++){
        //     pi += (ki+v)/(2*k*0.29+1.33);
        // }
        // pi *= 4;
        // d_dest_values[globalId] = (int)pi;

    #ifdef RECORDS
        if (isRecord)
            d_dest_keys[globalId] = d_source_keys[globalId];
    #endif
        globalId += globalSize;
    }

    // if (globalId == 0)  printf("total:%d\n",v);
}

#endif