#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//mem_read: repeat 100 times, unrolled
#define READ_REPEAT         (100)
#define WARP_SIZE            (32)
#define COALESCED_ADD(time)       (v += d_source_values[globalId + time]); 

#define STRIDED_ADD(time)   (v += d_source_values[begin + time]);

kernel void mem_read (
    global const TYPE* restrict d_source_values,
    global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    TYPE v = 0;

    //strided unrolling for Xeon & Xeon Phi
    // int warpId = globalId >> 4;
    // int begin = 16 * READ_REPEAT * warpId + (globalId & 0xf);   //the & needs a paranthesis!!
    // STRIDED_ADD( 0 ); STRIDED_ADD( 16 ); STRIDED_ADD( 32 );STRIDED_ADD( 48 );
    // STRIDED_ADD( 64 ); STRIDED_ADD( 80 ); STRIDED_ADD( 96 );STRIDED_ADD( 112 );
    // STRIDED_ADD( 128 ); STRIDED_ADD( 144 ); STRIDED_ADD( 160 );STRIDED_ADD( 176 );
    // STRIDED_ADD( 192 ); STRIDED_ADD( 208 ); STRIDED_ADD( 224 );STRIDED_ADD( 240 );
    // STRIDED_ADD( 256 ); STRIDED_ADD( 272 ); STRIDED_ADD( 288 );STRIDED_ADD( 304 );
    // STRIDED_ADD( 320 ); STRIDED_ADD( 336 ); STRIDED_ADD( 352 );STRIDED_ADD( 368 );
    // STRIDED_ADD( 384 ); STRIDED_ADD( 400 ); STRIDED_ADD( 416 );STRIDED_ADD( 432 );
    // STRIDED_ADD( 448 ); STRIDED_ADD( 464 ); STRIDED_ADD( 480 );STRIDED_ADD( 496 );
    // STRIDED_ADD( 512 ); STRIDED_ADD( 528 ); STRIDED_ADD( 544 );STRIDED_ADD( 560 );
    // STRIDED_ADD( 576 ); STRIDED_ADD( 592 ); STRIDED_ADD( 608 );STRIDED_ADD( 624 );
    // STRIDED_ADD( 640 ); STRIDED_ADD( 656 ); STRIDED_ADD( 672 );STRIDED_ADD( 688 );
    // STRIDED_ADD( 704 ); STRIDED_ADD( 720 ); STRIDED_ADD( 736 );STRIDED_ADD( 752 );
    // STRIDED_ADD( 768 ); STRIDED_ADD( 784 ); STRIDED_ADD( 800 );STRIDED_ADD( 816 );
    // STRIDED_ADD( 832 ); STRIDED_ADD( 848 ); STRIDED_ADD( 864 );STRIDED_ADD( 880 );
    // STRIDED_ADD( 896 ); STRIDED_ADD( 912 ); STRIDED_ADD( 928 );STRIDED_ADD( 944 );
    // STRIDED_ADD( 960 ); STRIDED_ADD( 976 ); STRIDED_ADD( 992 );STRIDED_ADD( 1008 );
    // STRIDED_ADD( 1024 ); STRIDED_ADD( 1040 ); STRIDED_ADD( 1056 );STRIDED_ADD( 1072 );
    // STRIDED_ADD( 1088 ); STRIDED_ADD( 1104 ); STRIDED_ADD( 1120 );STRIDED_ADD( 1136 );
    // STRIDED_ADD( 1152 ); STRIDED_ADD( 1168 ); STRIDED_ADD( 1184 );STRIDED_ADD( 1200 );
    // STRIDED_ADD( 1216 ); STRIDED_ADD( 1232 ); STRIDED_ADD( 1248 );STRIDED_ADD( 1264 );
    // STRIDED_ADD( 1280 ); STRIDED_ADD( 1296 ); STRIDED_ADD( 1312 );STRIDED_ADD( 1328 );
    // STRIDED_ADD( 1344 ); STRIDED_ADD( 1360 ); STRIDED_ADD( 1376 );STRIDED_ADD( 1392 );
    // STRIDED_ADD( 1408 ); STRIDED_ADD( 1424 ); STRIDED_ADD( 1440 );STRIDED_ADD( 1456 );
    // STRIDED_ADD( 1472 ); STRIDED_ADD( 1488 ); STRIDED_ADD( 1504 );STRIDED_ADD( 1520 );
    // STRIDED_ADD( 1536 ); STRIDED_ADD( 1552 ); STRIDED_ADD( 1568 );STRIDED_ADD( 1584 );
    // d_dest_values[globalId] = v;

    //coalesced unrolling for GPU
    int globalSize = get_global_size(0);
    int global_output = globalId;
    COALESCED_ADD( 0 ); COALESCED_ADD( 8388608 ); COALESCED_ADD( 16777216 );COALESCED_ADD( 25165824 );
    COALESCED_ADD( 33554432 ); COALESCED_ADD( 41943040 ); COALESCED_ADD( 50331648 );COALESCED_ADD( 58720256 );
    COALESCED_ADD( 67108864 ); COALESCED_ADD( 75497472 ); COALESCED_ADD( 83886080 );COALESCED_ADD( 92274688 );
    COALESCED_ADD( 100663296 ); COALESCED_ADD( 109051904 ); COALESCED_ADD( 117440512 );COALESCED_ADD( 125829120 );
    COALESCED_ADD( 134217728 ); COALESCED_ADD( 142606336 ); COALESCED_ADD( 150994944 );COALESCED_ADD( 159383552 );
    COALESCED_ADD( 167772160 ); COALESCED_ADD( 176160768 ); COALESCED_ADD( 184549376 );COALESCED_ADD( 192937984 );
    COALESCED_ADD( 201326592 ); COALESCED_ADD( 209715200 ); COALESCED_ADD( 218103808 );COALESCED_ADD( 226492416 );
    COALESCED_ADD( 234881024 ); COALESCED_ADD( 243269632 ); COALESCED_ADD( 251658240 );COALESCED_ADD( 260046848 );
    COALESCED_ADD( 268435456 ); COALESCED_ADD( 276824064 ); COALESCED_ADD( 285212672 );COALESCED_ADD( 293601280 );
    COALESCED_ADD( 301989888 ); COALESCED_ADD( 310378496 ); COALESCED_ADD( 318767104 );COALESCED_ADD( 327155712 );
    COALESCED_ADD( 335544320 ); COALESCED_ADD( 343932928 ); COALESCED_ADD( 352321536 );COALESCED_ADD( 360710144 );
    COALESCED_ADD( 369098752 ); COALESCED_ADD( 377487360 ); COALESCED_ADD( 385875968 );COALESCED_ADD( 394264576 );
    COALESCED_ADD( 402653184 ); COALESCED_ADD( 411041792 ); COALESCED_ADD( 419430400 );COALESCED_ADD( 427819008 );
    COALESCED_ADD( 436207616 ); COALESCED_ADD( 444596224 ); COALESCED_ADD( 452984832 );COALESCED_ADD( 461373440 );
    COALESCED_ADD( 469762048 ); COALESCED_ADD( 478150656 ); COALESCED_ADD( 486539264 );COALESCED_ADD( 494927872 );
    COALESCED_ADD( 503316480 ); COALESCED_ADD( 511705088 ); COALESCED_ADD( 520093696 );COALESCED_ADD( 528482304 );
    COALESCED_ADD( 536870912 ); COALESCED_ADD( 545259520 ); COALESCED_ADD( 553648128 );COALESCED_ADD( 562036736 );
    COALESCED_ADD( 570425344 ); COALESCED_ADD( 578813952 ); COALESCED_ADD( 587202560 );COALESCED_ADD( 595591168 );
    COALESCED_ADD( 603979776 ); COALESCED_ADD( 612368384 ); COALESCED_ADD( 620756992 );COALESCED_ADD( 629145600 );
    COALESCED_ADD( 637534208 ); COALESCED_ADD( 645922816 ); COALESCED_ADD( 654311424 );COALESCED_ADD( 662700032 );
    d_dest_values[global_output] = v;
}

kernel void mem_write (global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = 12345;
}

kernel void mem_mul (
    global const TYPE2* restrict d_source_values, 
    global TYPE2* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId] * 3;
}

kernel void mem_mul_coalesced (
    global const TYPE* restrict d_source_values, 
    global TYPE* restrict d_dest_values,
    const int repeat)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for(int i = 0; i < repeat; i++) {
        d_dest_values[globalId] = d_source_values[globalId] * 3;
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
        d_dest_values[begin+i] = d_source_values[begin+i] * 3 ;
    }
}

kernel void mem_mul_strided_warpwise (
    global const TYPE* restrict d_source_values, 
    global TYPE* restrict d_dest_values,
    const int repeat)
{
    //for Nvidia GPU, warpsize = 32, for Xeon & Xeon Phi, warpsiez = 16
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        d_dest_values[idx] = d_source_values[idx] * 3 ;
        idx += WARP_SIZE;
    }
}

#endif