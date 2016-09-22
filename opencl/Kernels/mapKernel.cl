#ifndef MAP_KERNEL_CL
#define MAP_KERNEL_CL

#define WARP_SIZE       (32)

#define RANGE           (536870912)     //max data size, unique

#define IF0             if (key == 0)   \
                            d_dest_keys[idx] = d_source_keys[idx];
#define ELSE(myKey)     else if (key == myKey)  \
                            d_dest_keys[idx] = d_source_keys[idx] + myKey;
#define ENDELSE(myKey)  else            \ 
                            d_dest_keys[idx] = d_source_keys[idx] + myKey;

//default localSize: 1024, gridSize: 8192, repeat:64, total size: 536870912

/*
 *  map1: get the last 2 bits of the keys to obtain the bucket idx
 */
kernel void map_hash (
    global const int* d_source_values, 
    global int* restrict buckets,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        buckets[idx] = d_source_values[idx] & 0x3;
        idx += WARP_SIZE;
    }
}

/*
 *  # of branches varies from 1 to 3
 *  datasize 536870912, value from 0 to 536870911
 */
kernel void map_branch_for (
    global const int* d_source_keys,
    global int* restrict d_dest_keys,
    const int repeat, const int branches)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {

        // implemented using for
        int key = d_source_keys[idx] % branches;
        for(int bra = branches - 1; bra >= 0 ; bra --) {
            if (key == bra) {
                d_dest_keys[idx] = d_source_keys[idx] + key;
                break;
            }
        }
        idx += WARP_SIZE;
    }
}

kernel void map_branch_1 (
    global const int* d_source_keys,
    global int* restrict d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {

        // universal branch
        int key = d_source_keys[idx] % 1;
        if (key == 0)   d_dest_keys[idx] = d_source_keys[idx] ;

        idx += WARP_SIZE;
    }
}

kernel void map_branch_2 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 2;
        IF0 ENDELSE(1)  //2 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_3 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 3;
        IF0 ELSE(1) ENDELSE(2)  // 3 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_4 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 4;
        IF0 ELSE(1) ELSE(2) ENDELSE(3)  // 4 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_5 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 5;
        IF0 ELSE(1) ELSE(2) ELSE(3) ENDELSE(4)  // 5 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_6 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 6;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ENDELSE(5)  // 6 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_7 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 7;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ENDELSE(6)  // 7 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_8 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 8;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ENDELSE(7)  // 8 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_9 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 9;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) ENDELSE(8)  // 9 branches
        idx += WARP_SIZE;
    }
}
kernel void map_branch_10 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 10;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) ELSE(8) ENDELSE(9)  // 10 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_11 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 11;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) ELSE(8) ELSE(9) ENDELSE(10)  // 11 branches
        idx += WARP_SIZE;
    }
}
kernel void map_branch_12 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 12;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) \
        ELSE(8) ELSE(9) ELSE(10) ENDELSE(11)  // 12 branches
        idx += WARP_SIZE;
    }
}
kernel void map_branch_13 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 13;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ENDELSE(12)  // 13 branches
        idx += WARP_SIZE;
    }
}
kernel void map_branch_14 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 14;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ENDELSE(13)  // 14 branches
        idx += WARP_SIZE;
    }
}
kernel void map_branch_15 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 15;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ENDELSE(14)  // 15 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_16 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 16;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7) \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14) ENDELSE(15)  // 16 branches
        idx += WARP_SIZE;
    }
}

kernel void map_branch_17 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 17;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ENDELSE(16)  
        idx += WARP_SIZE;
    }
}

kernel void map_branch_18 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 18;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ENDELSE(17) 
        idx += WARP_SIZE;
    }
}

kernel void map_branch_19 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 19;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ENDELSE(18)  
        idx += WARP_SIZE;
    }
}

kernel void map_branch_20 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 20;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ENDELSE(19)  
        idx += WARP_SIZE;
    }
}

kernel void map_branch_21 (
    global const int* d_source_keys,
    global int* restrict d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {

        // universal branch
        int key = d_source_keys[idx] % 21;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ENDELSE(20)  
        idx += WARP_SIZE;
    }
}

kernel void map_branch_22 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 22;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ENDELSE(21)  
        idx += WARP_SIZE;
    }
}

kernel void map_branch_23 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 23;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ENDELSE(22)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_24 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 24;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ENDELSE(23)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_25 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 25;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ENDELSE(24)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_26 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 26;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ENDELSE(25)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_27 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 27;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ENDELSE(26)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_28 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 28;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ENDELSE(27)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_29 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 29;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ENDELSE(28)
        idx += WARP_SIZE;
    }
}
kernel void map_branch_30 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 30;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ENDELSE(29)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_31 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 31;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ELSE(29) ENDELSE(30)
        idx += WARP_SIZE;
    }
}
kernel void map_branch_32 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 32;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ELSE(29) ELSE(30) ENDELSE(31)
        idx += WARP_SIZE;
    }
}
kernel void map_branch_33 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 33;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ELSE(29) ELSE(30) ELSE(31) ENDELSE(32)
        idx += WARP_SIZE;
    }
}
kernel void map_branch_34 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 34;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ELSE(29) ELSE(30) ELSE(31) ELSE(32) ENDELSE(33)
        idx += WARP_SIZE;
    }
}

kernel void map_branch_35 (
    global const int* d_source_keys,
    global int* d_dest_keys,
    const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        int key = d_source_keys[idx] % 35;
        IF0 ELSE(1) ELSE(2) ELSE(3) ELSE(4) ELSE(5) ELSE(6) ELSE(7)     \
        ELSE(8) ELSE(9) ELSE(10) ELSE(11) ELSE(12) ELSE(13) ELSE(14)    \
        ELSE(15) ELSE(16) ELSE(17) ELSE(18) ELSE(19) ELSE(20) ELSE(21)  \
        ELSE(22) ELSE(23) ELSE(24) ELSE(25) ELSE(26) ELSE(27) ELSE(28)  \
        ELSE(29) ELSE(30) ELSE(31) ELSE(32) ELSE(33) ENDELSE(34)
        idx += WARP_SIZE;
    }
}

//each thread processes 16 elements
kernel void map_trans_blank_float (
    global const float* alpha, const global float* beta, const float r,
    global float* restrict x, global float* restrict y, global float* restrict z, const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        x[idx] = r + alpha[idx] + beta[idx];
        y[idx] = r ;
        z[idx] = r ;
        idx += WARP_SIZE;
    }
}

kernel void map_trans_float (
    global const float* alpha, const global float* beta, const float r,
    global float* restrict x, global float* restrict y, global float* restrict z, const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        x[idx] = r * native_cos(alpha[idx]) * native_cos(beta[idx]);
        y[idx] = r * native_sin(alpha[idx]) * native_cos(beta[idx]);
        z[idx] = r * native_sin(beta[idx]);
        idx += WARP_SIZE;
    }
}

kernel void map_trans_blank_double (
    global const double* alpha, const global double* beta, const double r,
    global double* restrict x, global double* restrict y, global double* restrict z, const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        x[idx] = r + alpha[idx] + beta[idx];
        y[idx] = r ;
        z[idx] = r ;
        idx += WARP_SIZE;
    }
}

kernel void map_trans_double (
    global const double* alpha, const global double* beta, const double r,
    global double* restrict x, global double* restrict y, global double* restrict z, const int repeat)
{
    int globalId = get_global_id(0);
    int warp_num = globalId / WARP_SIZE;
    int idx = WARP_SIZE * repeat * warp_num + (globalId & (WARP_SIZE-1));

    for(int i = 0; i < repeat; i++) {
        x[idx] = r * native_cos(alpha[idx]) * native_cos(beta[idx]);
        y[idx] = r * native_sin(alpha[idx]) * native_cos(beta[idx]);
        z[idx] = r * native_sin(beta[idx]);
        idx += WARP_SIZE;
    }
}

#endif