#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

#include "params.h"

#ifdef KVS_AOS
    typedef int2 Tuple;  /*for AOS*/
    #define GET_X_VALUE(d_in, idx)    d_in[idx].x
#else
    typedef int Tuple;    /*for KO*/
    #define GET_X_VALUE(d_in, idx)    d_in[idx]
#endif

#ifdef SMALLER_WARP_SIZE        //num <= WARP_SIZE
    #define LOCAL_SCAN(arr,num,offset)                                      \
    if (local_id < num) {                                                    \
        int temp = arr[local_id+offset];                                     \
        if (local_id >= 1) arr[local_id+offset] += arr[local_id - 1+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (local_id >= 2) arr[local_id+offset] += arr[local_id - 2+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (local_id >= 4) arr[local_id+offset] += arr[local_id - 4+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (local_id >= 8) arr[local_id+offset] += arr[local_id - 8+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (local_id >= 16) arr[local_id+offset] += arr[local_id - 16+offset]; \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        arr[local_id+offset] -= temp;                                        \
    }                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);
#elif defined(LARGER_WARP_SIZE_SINGLE_LOOP)   // WARP_SIZE < num < WARP_SIZE_2
   #define LOCAL_SCAN(arr,num,offset)                                       \
   int lane = local_id & (WARP_SIZE-1);                                      \
   int warpId = local_id >> WARP_BITS;                                       \
   int warpNum = local_size >> WARP_BITS;                                    \
                                                                            \
   local int tempSums[WARP_SIZE];                                           \
   if (local_id < num) {                                                     \
       int temp = arr[local_id+offset];                                      \
       if (lane >= 1) arr[local_id+offset] += arr[local_id - 1 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 2) arr[local_id+offset] += arr[local_id - 2 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 4) arr[local_id+offset] += arr[local_id - 4 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 8) arr[local_id+offset] += arr[local_id - 8 +offset];      \
       if (lane >= 16) arr[local_id+offset] += arr[local_id - 16 +offset];    \
       if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];\
       arr[local_id+offset] -= temp;                                         \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);                                            \
                                                                            \
   if (warpId == 0) {                                                       \
       int temp = tempSums[local_id];                                        \
       if (lane >= 1) tempSums[local_id] += tempSums[local_id - 1];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 2) tempSums[local_id] += tempSums[local_id - 2];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 4) tempSums[local_id] += tempSums[local_id - 4];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 8) tempSums[local_id] += tempSums[local_id - 8];           \
       if (lane >= 16) tempSums[local_id] += tempSums[local_id - 16];         \
       tempSums[local_id] -= temp;                                           \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);                                            \
                                                                            \
   if (local_id < num) {                                                     \
       arr[local_id+offset] += tempSums[warpId];                             \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);
#elif defined(LARGER_WARP_SIZE_MULTIPLE_LOOPS)
    #define LOCAL_SCAN(arr,num,offset)                                      \
    int lane = local_id & (WARP_SIZE-1);                                     \
    int warpId = local_id >> WARP_BITS;                                      \
    int warpNum = local_size >> WARP_BITS;                                   \
                                                                            \
    local int tempSums[WARP_SIZE];                                          \
    int myPrivate[LOOPS];                                                   \
                                                                            \
    for(int i = 0; i < LOOPS; i++)                                          \
        myPrivate[i] = arr[local_id*LOOPS + i + offset];                     \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    int local_temp0 = myPrivate[0];                                         \
    myPrivate[0] = 0;                                                       \
    for(int r = 1; r < LOOPS; r++) {                                        \
        int local_temp1 = myPrivate[r];                                     \
        myPrivate[r] = local_temp0 + myPrivate[r-1];                        \
        local_temp0 = local_temp1;                                          \
    }                                                                       \
    int temp0 = local_temp0 + myPrivate[LOOPS-1];                           \
    arr[local_id+offset] = temp0;                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    if (lane >= 1) arr[local_id+offset] += arr[local_id - 1 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 2) arr[local_id+offset] += arr[local_id - 2 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 4) arr[local_id+offset] += arr[local_id - 4 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 8) arr[local_id+offset] += arr[local_id - 8 +offset];         \
    if (lane >= 16) arr[local_id+offset] += arr[local_id - 16 +offset];       \
    if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];   \
    arr[local_id+offset] -= temp0;                                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    if (warpId == 0) {                                                      \
        int temp = tempSums[local_id];                                       \
        if (lane >= 1) tempSums[local_id] += tempSums[local_id - 1];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 2) tempSums[local_id] += tempSums[local_id - 2];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 4) tempSums[local_id] += tempSums[local_id - 4];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 8) tempSums[local_id] += tempSums[local_id - 8];          \
        if (lane >= 16) tempSums[local_id] += tempSums[local_id - 16];        \
        tempSums[local_id] -= temp;                                          \
    }                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    int myLocalSum = arr[local_id+offset] + tempSums[warpId];                \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    for(int i = 0; i < LOOPS; i++)                                          \
        arr[local_id*LOOPS + i + offset] = myPrivate[i] + myLocalSum;        \
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define LOCAL_SCAN(arr,num,offset) ;
#endif

int findLog2(int input) {
    int lookup[21] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576};
    int start = 0, end = 21, middle = (start+end)/2;

    while(lookup[middle] != input) {
        if (start >= end)   return -1;
        if (input > lookup[middle])  start = middle+1;
        else                         end = middle-1;
        middle = (start+end)/2;
    }
    return middle;
}

/*local_size must be the power of 2*/
void compute_mixed_access(
        unsigned step, unsigned global_id, unsigned global_size, unsigned len_total,
        unsigned *begin, unsigned *end)
{
    int step_log = findLog2(step);
    int tile = (len_total + global_size - 1) / global_size;

    int warp_id = global_id >> step_log;
    *begin = warp_id * step * tile + (global_id & (step-1));
    *end = (warp_id + 1) * step * tile;
    if ((*end) > len_total)    *end = len_total;
}

//# buckets > local_size, each thread processes (loops) elements
// void temp_scan(local int* arr, int num, int offset, global int* help, global int* help2) {
//     const int local_id = get_local_id(0);
//     const int local_size = get_local_size(0);
//     const int lane = local_id & (WARP_SIZE-1);
//     int warpId = local_id >> WARP_BITS;
//     int warpNum = local_size >> WARP_BITS;
//     int loops = (num + local_size - 1)/local_size;
//
//     local int tempSums[WARP_SIZE];
//     int myPrivate[3];
//
//     //local to registers
//     for(int i = 0; i < loops; i++) myPrivate[i] = arr[local_id*loops + i + offset];
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     //register scan
//     int local_temp0 = myPrivate[0];
//     myPrivate[0] = 0;
//     for(int r = 1; r < loops; r++) {
//         int local_temp1 = myPrivate[r];
//         myPrivate[r] = local_temp0 + myPrivate[r-1];
//         local_temp0 = local_temp1;
//     }
//     int temp0 = local_temp0 + myPrivate[loops-1];
//     arr[local_id+offset] = temp0;
//     barrier(CLK_LOCAL_MEM_FENCE);
//
// int b = get_group_id(0);
// if (b==0)   help[local_id] = arr[local_id];
// barrier(CLK_LOCAL_MEM_FENCE);
//
//
//     //now arr has only (local_size) elements
//     if (lane >= 1) arr[local_id+offset] += arr[local_id - 1 +offset];
//     if (lane >= 2) arr[local_id+offset] += arr[local_id - 2 +offset];
//     if (lane >= 4) arr[local_id+offset] += arr[local_id - 4 +offset];
//     if (lane >= 8) arr[local_id+offset] += arr[local_id - 8 +offset];
//     if (lane >= 16) arr[local_id+offset] += arr[local_id - 16 +offset];
//     if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];
//     arr[local_id+offset] -= temp0;
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     if (warpId == 0) {
//         int temp = tempSums[local_id];
//         if (lane >= 1) tempSums[local_id] += tempSums[local_id - 1];
//         if (lane >= 2) tempSums[local_id] += tempSums[local_id - 2];
//         if (lane >= 4) tempSums[local_id] += tempSums[local_id - 4];
//         if (lane >= 8) tempSums[local_id] += tempSums[local_id - 8];
//         if (lane >= 16) tempSums[local_id] += tempSums[local_id - 16];
//         tempSums[local_id] -= temp;
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     int myLocalSum = arr[local_id+offset] + tempSums[warpId];
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     //put registers data back to the local memory
//     for(int i = 0; i < loops; i++)
//         arr[local_id*loops + i + offset] = myPrivate[i] + myLocalSum;
//     barrier(CLK_LOCAL_MEM_FENCE);
// }

//local sklansky scan
void sklansky_scan(local int* lo, int length)
{
    int local_id = get_local_id(0);

    int mask_j = (length>>1) -1;
    int mask_k = 0;
    int temp = 1;
    int localTemp;

    if (local_id < length)
        localTemp = lo[local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    int length_log = findLog2(length);
    for(int i = 0; i < length_log; i++) {
        if (local_id < (length>>1)) {            //only half of the threads execute
            int para_j = (local_id >> i) & mask_j;
            int para_k = local_id & mask_k;

            int j = temp - 1 + (temp<<1)*para_j;
            int k = para_k;
            lo[j+k+1] = lo[j] + lo[j+k+1];

            mask_j >>= 1;
            mask_k = (mask_k<<1)+1;
            temp <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id < length)
        lo[local_id] -= localTemp;
    barrier(CLK_LOCAL_MEM_FENCE);
}

void scan_local(local int* lo, int length)
{
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    int tempStore;  //to store the first 1024 elements in the original lo array
    int tempSum;
    int row_start, row_end;
    local int temps[64];
//    length is smaller than the local_size, just use the local scan scheme
    if(length <= local_size) {
        sklansky_scan(lo, length);
        return;
    }

    //1. Reduction (per thread)
    int ele_per_thread = (length + local_size - 1)/local_size;
    row_start = local_id * ele_per_thread;
    row_end = (local_id + 1) * ele_per_thread;

    int local_sum = 0;
    for (int r = row_start; r < row_end; r++) {
        local_sum += lo[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    tempStore = lo[local_id];                    //switch out the first local_size elements
    lo[local_id] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //local scan
    sklansky_scan(lo, local_size);

    tempSum = lo[local_id];
    lo[local_id] = tempStore;    //switch back the first local_size elements

    barrier(CLK_LOCAL_MEM_FENCE);

    //3. Scan
    int local_temp0 = lo[row_start];
    lo[row_start] = tempSum;
    for (int r = row_start + 1; r < row_end; r++) {
        int local_temp1 = lo[r];
        lo[r] = local_temp0 + lo[r - 1];
        local_temp0 = local_temp1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//single-threaded local exclusive scan
void scan_local_single(local int* lo, int length) {
    int local_id = get_local_id(0);
    if (local_id == 0) {
        int acc = 0;
        for(int i = 0; i <length; i++) {
            int temp = lo[i];
            lo[i] = acc;
            acc += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*gather the start position of each partition (optional)*/
kernel void gatherStartPos( global const int *d_his,
                            int his_len,
                            global int *d_start,
                            int gridSizeUsedInHis)
{
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    while (global_id * gridSizeUsedInHis < his_len) {
        d_start[global_id] = d_his[global_id*gridSizeUsedInHis];
        global_id += global_size;
    }
}

/*WI-level histogram: each thraed has a private histogram stored in the local memory*/
kernel void WI_histogram(
        global const Tuple *d_in,       /*input keys*/
        int len_total,                  /*length of the dataset*/
        global int *his,                /*output histogram*/
        int buckets,
        local int *local_buckets)
{
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global, begin_local, end_local;

    compute_mixed_access(
            step, local_id, local_size, buckets * local_size,
            &begin_local, &end_local);
    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    for(int i = begin_local; i < end_local; i += step)
        local_buckets[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = begin_global; i < end_global; i += step) {
        offset = GET_X_VALUE(d_in, i) & mask;
        local_buckets[offset*local_size+local_id]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*histogram write*/
    for(int i = 0; i < buckets; i++)
        his[i*global_size+global_id] = local_buckets[i*local_size+local_id];
}

kernel void WI_shuffle(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    int len_total,
    global int *his,
    int buckets,
    local int *local_buckets)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global;

    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    for(int i = 0; i < buckets; i++)
        local_buckets[i*local_size + local_id] = his[i*global_size+global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = begin_global; i < end_global; i += step) {
        offset = GET_X_VALUE(d_in, i) & mask;
        int idx = offset*local_size + local_id;

        d_out[local_buckets[idx]] = d_in[i];
#ifdef KVS_SOA
        d_out_values[local_buckets[idx]] = d_in_values[i];
#endif
        local_buckets[idx]++;
    }
}

/*------------ WG-level kernels : WIs in a WG share a histogram ------------*/
kernel void WG_histogram(
    global const Tuple *d_in,   /*input data*/
    int len_total,              /*length of the dataset*/
    global int *his,            /*histogram output*/
    local int* local_buc,       /*local buffer: buckets*sizeof(int)*/
    int buckets)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global;

    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    /*local histogram initialization*/
    for(int i = local_id; i < buckets; i += local_size)
        local_buc[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*global sequential access*/
    for(int i = begin_global; i < end_global; i += step) {
       offset = GET_X_VALUE(d_in, i) & mask;
       atomic_inc(local_buc+offset);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*histogram write*/
    for(int i = local_id; i < buckets; i += local_size)
        his[i*num_groups+group_id] = local_buc[i];
}

kernel void WG_shuffle(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    int len_total,
    int buckets,
    global int *his,
    local int *local_buc)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global;

    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    for(int i = local_id; i < buckets; i += local_size)
        local_buc[i] = his[i*num_groups+group_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*global sequential access*/
    for(int i = begin_global; i < end_global; i += step) {
        offset = GET_X_VALUE(d_in, i) & mask;
        int pos = atomic_inc(local_buc+offset);
        d_out[pos] = d_in[i];
#ifdef KVS_SOA
        d_out_values[pos] = d_in_values[i];
#endif
    }
}

/*block-level split on key-value data with data reordering*/
kernel void WG_shuffle_varied(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    const int len_total,
    const int buckets,
    global int *his_scanned,     /*scanned histogram*/
    local int* local_start_ptrs, /*start pos of each bucket in local mem, bucket elements*/
    global int *his_origin,      /*original histogram*/
    local Tuple *reorder_buffer  /*tuple buffer for AOS*/
#ifdef KVS_SOA
    ,local Tuple *reorder_buffer_values   /*value buffer for SOA*/
#endif
)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global;

    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    /*load the scanned histogram*/
    local_start_ptrs[0] = 0;
    for(int i = local_id; i < buckets; i += local_size)
        local_start_ptrs[i+1] = his_origin[i*num_groups+group_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*scan the local ptrs exclusively*/
//    LOCAL_SCAN(local_start_ptrs, buckets,1);
    scan_local(local_start_ptrs+1, buckets);

    /*scatter the input to the local memory*/
    for(int i = begin_global; i < end_global; i += step) {
        offset = GET_X_VALUE(d_in, i) & mask;
        int acc = atomic_inc(local_start_ptrs+offset+1);

        /*write to the buffer*/
        reorder_buffer[acc] = d_in[i];
#ifdef KVS_SOA
        reorder_buffer_values[acc] = d_in_values[i];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_id; i < buckets; i += local_size)
        local_start_ptrs[i] = his_scanned[i*num_groups+group_id] - local_start_ptrs[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the data from the local mem to global mem (coalesced)
    int local_sum = local_start_ptrs[buckets];
    for(int i = local_id; i < local_sum; i += local_size) {
        offset = GET_X_VALUE(reorder_buffer, i) & mask;
        d_out[i+local_start_ptrs[offset]] = reorder_buffer[i];
#ifdef KVS_SOA
        d_out_values[i+local_start_ptrs[offset]] = reorder_buffer_values[i];
#endif
    }
}

#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE (64)     /*in bytes*/
#endif

#define ELE_PER_CACHELINE   (CACHELINE_SIZE/sizeof(Tuple))

kernel  __attribute__((work_group_size_hint(1, 1, 1)))
void WG_shuffle_fixed(
        global const Tuple *d_in,
        global Tuple *d_out,
#ifdef KVS_SOA
        global const Tuple *d_in_values,
        global Tuple *d_out_values,
#endif
        const int len_total,                /*len of the whole array*/
        const int buckets,                  /*number of buckets*/
        global int *his,                    /*scanned histogram*/
        local int *local_buc_ptr,           /*scanned local buckets ptrs: buckets*sizeof(int)*/
        global Tuple *reorder_buffer_all    /*tuple buffer for AOS*/
#ifdef KVS_SOA
        ,global Tuple *reorder_buffer_all_values   /*value buffer for SOA*/
#endif
)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned step = (local_size < WARP_SIZE) ? local_size : WARP_SIZE;
    unsigned mask = buckets - 1;
    unsigned offset, begin_global, end_global, begin_local, end_local, buffer_len;

    compute_mixed_access(
            step, local_id, local_size, buckets,
            &begin_local, &end_local);
    compute_mixed_access(
            step, global_id, global_size, len_total,
            &begin_global, &end_global);

    /*local buffer size*/
    global Tuple *local_buffer, *local_buffer_values;
    local_buffer = (global Tuple*)(reorder_buffer_all+ group_id*ELE_PER_CACHELINE*buckets);
#ifdef KVS_SOA
    local_buffer_values = (global Tuple*)(reorder_buffer_all_values+ group_id*ELE_PER_CACHELINE*buckets);
#endif

    /*load the scanned histogram and initialize the local buffer*/
    for(int i = begin_local; i < end_local; i += step) {
        local_buc_ptr[i] = his[i * num_groups + group_id];

        /*last element in the cacheline records the len*/
        GET_X_VALUE(local_buffer, (i+1)*ELE_PER_CACHELINE-1) = 0;
    }

    /*iterate the data partition*/
    for(int i = begin_global; i <end_global; i += step) {
        offset = GET_X_VALUE(d_in, i) & mask;
        unsigned buffer_len_idx = (offset+1)*ELE_PER_CACHELINE-1;

        /*write to the cache buffer*/
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);
        local_buffer[offset*ELE_PER_CACHELINE+buffer_len] = d_in[i];
#ifdef KVS_SOA
        local_buffer_values[offset*ELE_PER_CACHELINE+buffer_len] = d_in_values[i];
#endif

        /*update the counter*/
        GET_X_VALUE(local_buffer,buffer_len_idx)++;
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);

        if (buffer_len == (ELE_PER_CACHELINE-1)) {
            /*write the buffer to the global array*/
            for(int c = 0; c < ELE_PER_CACHELINE-1; c++) {
                d_out[local_buc_ptr[offset] + c] = local_buffer[offset * ELE_PER_CACHELINE + c];
#ifdef KVS_SOA
                d_out_values[local_buc_ptr[offset] + c] = local_buffer_values[offset * ELE_PER_CACHELINE + c];
#endif
            }
            local_buc_ptr[offset] += (ELE_PER_CACHELINE-1);
            GET_X_VALUE(local_buffer,buffer_len_idx) = 0;
        }
    }

    /*write the rest elements to the global array*/
    for(int i = begin_local; i < end_local; i += step) {
        unsigned buffer_len_idx = (i+1)*ELE_PER_CACHELINE-1;
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);
        for(int c = 0; c < buffer_len; c++) {
            d_out[local_buc_ptr[i]+c] = local_buffer[i*ELE_PER_CACHELINE+c];
#ifdef KVS_SOA
            d_out_values[local_buc_ptr[i]+c] = local_buffer_values[i*ELE_PER_CACHELINE+c];
#endif
        }
    }
}

/*---------------------- dedicated single-threaded kernels for CPUs and MICs --------------------------*/
/*
 * All the kernels are invoked with local_size=1
 * Only support KO and AOS
 */
kernel  __attribute__((work_group_size_hint(1, 1, 1)))
void single_histogram(
        global const Tuple *d_in,   /*input keys*/
        const int len_per_group,        /*elements processed by each WG*/
        const int len_total,            /*len of the whole array*/
        const int buckets,              /*number of buckets*/
        global int *his,                /*histogram output*/
        local int *local_buc)           /*local buckets: buckets*sizeof(int)*/
{
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset;
    unsigned start = len_per_group * group_id;
    unsigned end = len_per_group * (group_id+1);
    if (end > len_total)    end = len_total;

    /*local histogram initialization*/
    for(int i = 0; i < buckets; i++)    local_buc[i] = 0;

    /*iterate the data partition*/
    for(int i = start; i <end; i++) {
        offset = GET_X_VALUE(d_in, i) & mask;
        local_buc[offset]++;
    }

    /*output to the global histogram*/
    for(int i = 0; i < buckets; i++)
        his[i*num_groups+group_id] = local_buc[i];
}

kernel  __attribute__((work_group_size_hint(1, 1, 1)))
void single_shuffle(
        global const Tuple *d_in,
        global Tuple *d_out,
const int len_per_group,        /*elements processed by each WG*/
const int len_total,            /*len of the whole array*/
const int buckets,
        global int *his,                /*scanned histogram*/
        local int *local_buc)            /*scanned local buckets: buckets*sizeof(int)*/
{
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset;
    unsigned start = len_per_group * group_id;
    unsigned end = len_per_group * (group_id+1);
    if (end > len_total)    end = len_total;

    /*load the scanned histogram*/
    for(int i = 0; i < buckets; i++)
        local_buc[i] = his[i*num_groups+group_id];

    /*iterate the data partition*/
    for(int i = start; i <end; i++) {
        offset = GET_X_VALUE(d_in, i) & mask;
        unsigned addr = local_buc[offset]++;
        d_out[addr] = d_in[i];
    }
}

kernel  __attribute__((work_group_size_hint(1, 1, 1)))
void single_fixed_shuffle(
        global const Tuple *d_in,
        global Tuple *d_out,
        const int len_per_group,            /*elements processed by each WG*/
        const int len_total,                /*len of the whole array*/
        int buckets,                        /*number of buckets*/
        global int *his,                    /*scanned histogram*/
        local int *local_buc_ptr,           /*scanned local buckets ptrs: buckets*sizeof(int)*/
        global Tuple *reorder_buffer_all)    /*tuple buffer for AOS*/
{
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset, buffer_len;
    unsigned start = len_per_group * group_id;
    unsigned end = len_per_group * (group_id+1);
    if (end > len_total)    end = len_total;

    /*local buffer size*/
    global Tuple *local_buffer;
    local_buffer = (global Tuple*)(reorder_buffer_all+ group_id*ELE_PER_CACHELINE*buckets);

    /*load the scanned histogram and initialize the local buffer*/
    for(int i = 0; i < buckets; i++) {
        local_buc_ptr[i] = his[i * num_groups + group_id];

        /*last element in the cacheline records the len*/
        GET_X_VALUE(local_buffer, (i+1)*ELE_PER_CACHELINE-1) = 0;
    }

    /*iterate the data partition*/
    for(int i = start; i <end; i++) {
        offset = GET_X_VALUE(d_in, i) & mask;
        unsigned buffer_len_idx = (offset+1)*ELE_PER_CACHELINE-1;

        /*write to the cache buffer*/
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);
        local_buffer[offset*ELE_PER_CACHELINE+buffer_len] = d_in[i];

        /*update the counter*/
        GET_X_VALUE(local_buffer,buffer_len_idx)++;
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);

        if (buffer_len == (ELE_PER_CACHELINE-1)) {
            /*write the buffer to the global array*/
            for(int c = 0; c < ELE_PER_CACHELINE-1; c++) {
                d_out[local_buc_ptr[offset] + c] = local_buffer[offset * ELE_PER_CACHELINE + c];
            }
            local_buc_ptr[offset] += (ELE_PER_CACHELINE-1);
            GET_X_VALUE(local_buffer,buffer_len_idx) = 0;
        }
    }

    /*write the rest elements to the global array*/
    for(int i = 0; i < buckets; i++) {
        unsigned buffer_len_idx = (i+1)*ELE_PER_CACHELINE-1;
        buffer_len = GET_X_VALUE(local_buffer,buffer_len_idx);
        for(int c = 0; c < buffer_len; c++) {
            d_out[local_buc_ptr[i]+c] = local_buffer[i*ELE_PER_CACHELINE+c];
        }
    }
}

#endif