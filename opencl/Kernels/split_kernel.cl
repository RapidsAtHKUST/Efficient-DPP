#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

#ifdef KVS_AOS
    typedef int2 Tuple;  /*for AOS*/
    #define GET_X_VALUE(d_in, idx)    d_in[idx].x
#else
    typedef int Tuple;    /*for KO*/
    #define GET_X_VALUE(d_in, idx)    d_in[idx]
#endif

#define WARP_BITS   (4)
#define WARP_SIZE   (1<<WARP_BITS)

#ifdef SMALLER_WARP_SIZE        //num <= WARP_SIZE
    #define LOCAL_SCAN(arr,num,offset)                                      \
    if (localId < num) {                                                    \
        int temp = arr[localId+offset];                                     \
        if (localId >= 1) arr[localId+offset] += arr[localId - 1+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (localId >= 2) arr[localId+offset] += arr[localId - 2+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (localId >= 4) arr[localId+offset] += arr[localId - 4+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (localId >= 8) arr[localId+offset] += arr[localId - 8+offset];   \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (localId >= 16) arr[localId+offset] += arr[localId - 16+offset]; \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        arr[localId+offset] -= temp;                                        \
    }                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);
#elif defined(LARGER_WARP_SIZE_SINGLE_LOOP)   // WARP_SIZE < num < WARP_SIZE_2
   #define LOCAL_SCAN(arr,num,offset)                                       \
   int lane = localId & (WARP_SIZE-1);                                      \
   int warpId = localId >> WARP_BITS;                                       \
   int warpNum = localSize >> WARP_BITS;                                    \
                                                                            \
   local int tempSums[WARP_SIZE];                                           \
   if (localId < num) {                                                     \
       int temp = arr[localId+offset];                                      \
       if (lane >= 1) arr[localId+offset] += arr[localId - 1 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];      \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 8) arr[localId+offset] += arr[localId - 8 +offset];      \
       if (lane >= 16) arr[localId+offset] += arr[localId - 16 +offset];    \
       if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];\
       arr[localId+offset] -= temp;                                         \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);                                            \
                                                                            \
   if (warpId == 0) {                                                       \
       int temp = tempSums[localId];                                        \
       if (lane >= 1) tempSums[localId] += tempSums[localId - 1];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 2) tempSums[localId] += tempSums[localId - 2];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 4) tempSums[localId] += tempSums[localId - 4];           \
       mem_fence(CLK_LOCAL_MEM_FENCE); \
       if (lane >= 8) tempSums[localId] += tempSums[localId - 8];           \
       if (lane >= 16) tempSums[localId] += tempSums[localId - 16];         \
       tempSums[localId] -= temp;                                           \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);                                            \
                                                                            \
   if (localId < num) {                                                     \
       arr[localId+offset] += tempSums[warpId];                             \
   }                                                                        \
   barrier(CLK_LOCAL_MEM_FENCE);
#elif defined(LARGER_WARP_SIZE_MULTIPLE_LOOPS)
    #define LOCAL_SCAN(arr,num,offset)                                      \
    int lane = localId & (WARP_SIZE-1);                                     \
    int warpId = localId >> WARP_BITS;                                      \
    int warpNum = localSize >> WARP_BITS;                                   \
                                                                            \
    local int tempSums[WARP_SIZE];                                          \
    int myPrivate[LOOPS];                                                   \
                                                                            \
    for(int i = 0; i < LOOPS; i++)                                          \
        myPrivate[i] = arr[localId*LOOPS + i + offset];                     \
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
    arr[localId+offset] = temp0;                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    if (lane >= 1) arr[localId+offset] += arr[localId - 1 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];         \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    if (lane >= 8) arr[localId+offset] += arr[localId - 8 +offset];         \
    if (lane >= 16) arr[localId+offset] += arr[localId - 16 +offset];       \
    if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];   \
    arr[localId+offset] -= temp0;                                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    if (warpId == 0) {                                                      \
        int temp = tempSums[localId];                                       \
        if (lane >= 1) tempSums[localId] += tempSums[localId - 1];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 2) tempSums[localId] += tempSums[localId - 2];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 4) tempSums[localId] += tempSums[localId - 4];          \
        mem_fence(CLK_LOCAL_MEM_FENCE); \
        if (lane >= 8) tempSums[localId] += tempSums[localId - 8];          \
        if (lane >= 16) tempSums[localId] += tempSums[localId - 16];        \
        tempSums[localId] -= temp;                                          \
    }                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    int myLocalSum = arr[localId+offset] + tempSums[warpId];                \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    for(int i = 0; i < LOOPS; i++)                                          \
        arr[localId*LOOPS + i + offset] = myPrivate[i] + myLocalSum;        \
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define LOCAL_SCAN(arr,num,offset) ;
#endif

int findLog2(int input) {
    int lookup[20] = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576};
    int start = 0, end = 20, middle = (start+end)/2;

    while(lookup[middle] != input) {
        if (start >= end)   return -1;
        if (input > lookup[middle])  start = middle+1;
        else                         end = middle-1;
        middle = (start+end)/2;
    }
    return middle+1;
}

//# buckets > localSize, each thread processes (loops) elements
// void temp_scan(local int* arr, int num, int offset, global int* help, global int* help2) {
//     const int localId = get_local_id(0);
//     const int localSize = get_local_size(0);
//     const int lane = localId & (WARP_SIZE-1);
//     int warpId = localId >> WARP_BITS;
//     int warpNum = localSize >> WARP_BITS;
//     int loops = (num + localSize - 1)/localSize;
//
//     local int tempSums[WARP_SIZE];
//     int myPrivate[3];
//
//     //local to registers
//     for(int i = 0; i < loops; i++) myPrivate[i] = arr[localId*loops + i + offset];
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
//     arr[localId+offset] = temp0;
//     barrier(CLK_LOCAL_MEM_FENCE);
//
// int b = get_group_id(0);
// if (b==0)   help[localId] = arr[localId];
// barrier(CLK_LOCAL_MEM_FENCE);
//
//
//     //now arr has only (localSize) elements
//     if (lane >= 1) arr[localId+offset] += arr[localId - 1 +offset];
//     if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];
//     if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];
//     if (lane >= 8) arr[localId+offset] += arr[localId - 8 +offset];
//     if (lane >= 16) arr[localId+offset] += arr[localId - 16 +offset];
//     if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];
//     arr[localId+offset] -= temp0;
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     if (warpId == 0) {
//         int temp = tempSums[localId];
//         if (lane >= 1) tempSums[localId] += tempSums[localId - 1];
//         if (lane >= 2) tempSums[localId] += tempSums[localId - 2];
//         if (lane >= 4) tempSums[localId] += tempSums[localId - 4];
//         if (lane >= 8) tempSums[localId] += tempSums[localId - 8];
//         if (lane >= 16) tempSums[localId] += tempSums[localId - 16];
//         tempSums[localId] -= temp;
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     int myLocalSum = arr[localId+offset] + tempSums[warpId];
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     //put registers data back to the local memory
//     for(int i = 0; i < loops; i++)
//         arr[localId*loops + i + offset] = myPrivate[i] + myLocalSum;
//     barrier(CLK_LOCAL_MEM_FENCE);
// }

//local sklansky scan
void sklansky_scan(local int* lo, int length)
{
    int localId = get_local_id(0);

    int mask_j = (length>>1) -1;
    int mask_k = 0;
    int temp = 1;
    int localTemp;

    if (localId < length)
        localTemp = lo[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    int length_log = findLog2(length);
    for(int i = 0; i < length_log; i++) {
        if (localId < (length>>1)) {            //only half of the threads execute
            int para_j = (localId >> i) & mask_j;
            int para_k = localId & mask_k;

            int j = temp - 1 + (temp<<1)*para_j;
            int k = para_k;
            lo[j+k+1] = lo[j] + lo[j+k+1];

            mask_j >>= 1;
            mask_k = (mask_k<<1)+1;
            temp <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId < length)
        lo[localId] -= localTemp;
    barrier(CLK_LOCAL_MEM_FENCE);
}

void scan_local(local int* lo, int length)
{
    const int localId = get_local_id(0);
    const int localSize = get_local_size(0);
    int tempStore;  //to store the first 1024 elements in the original lo array
    int tempSum;
    int row_start, row_end;
    local int temps[64];
//    length is smaller than the localSize, just use the local scan scheme
    if(length <= localSize) {
        sklansky_scan(lo, length);
        return;
    }

    //1. Reduction (per thread)
    int ele_per_thread = (length + localSize - 1)/localSize;
    row_start = localId * ele_per_thread;
    row_end = (localId + 1) * ele_per_thread;

    int local_sum = 0;
    for (int r = row_start; r < row_end; r++) {
        local_sum += lo[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    tempStore = lo[localId];                    //switch out the first localSize elements
    lo[localId] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //local scan
    sklansky_scan(lo, localSize);

    tempSum = lo[localId];
    lo[localId] = tempStore;    //switch back the first localSize elements

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
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId * gridSizeUsedInHis < his_len) {
        d_start[globalId] = d_his[globalId*gridSizeUsedInHis];
        globalId += globalSize;
    }
}

/*WI-level histogram: each thraed has a private histogram stored in the local memory*/
kernel void WI_histogram(
        global const Tuple *d_in,   /*input keys*/
        int length,                 /*length of the dataset*/
        global int *his,            /*output histogram*/
        int buckets,
        local int *local_buckets)
{
    int globalId = get_global_id(0);
    int globalId_fixed = globalId;
    int globalSize = get_global_size(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    unsigned mask = buckets - 1;
    unsigned offset;

    int idx = localId;
    while (idx < buckets * localSize) {
        local_buckets[idx] = 0;
        idx += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (globalId < length) {
        offset = GET_X_VALUE(d_in, globalId) & mask;
        local_buckets[offset*localSize+localId]++;
        globalId += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the histogram back to the global memory
    for(int i = 0; i < buckets; i++) {
        his[i*globalSize+globalId_fixed] = local_buckets[i*localSize+localId];
    }
}

kernel void WI_scatter(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    int length,
    global int *his,
    int buckets,
    local int *local_buckets)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    unsigned mask = buckets - 1;
    unsigned offset;

    for(int i = 0; i < buckets; i++) {
        local_buckets[i*localSize + localId] = his[i*globalSize+globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (globalId < length) {
        offset = GET_X_VALUE(d_in, globalId) & mask;
        int idx = offset*localSize + localId;

        d_out[local_buckets[idx]] = d_in[globalId];
#ifdef KVS_SOA
        d_out_values[local_buckets[idx]] = d_in_values[globalId];
#endif
        local_buckets[idx]++;
        globalId += globalSize;
    }
}

/*------------ WG-level kernels : WIs in a WG share a histogram ------------*/
kernel void WG_histogram(
    global const Tuple *d_in,   /*input data*/
    int length,                 /*length of the dataset*/
    global int *his,            /*histogram output*/
    local int* local_his,       /*local buffer: buckets*sizeof(int)*/
    int buckets)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset;

    //local histogram initialization
    int i = localId;
    while (i < buckets) {
        local_his[i] = 0;
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = globalId;
    while (i < length) {
        offset = GET_X_VALUE(d_in, i) & mask;
        atomic_inc(local_his+offset);
        i += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = localId;
    while (i < buckets) {
        his[i*groupNum+groupId] = local_his[i];
        i += localSize;
    }
}

kernel void WG_scatter(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    int length,
    int buckets,
    global int *his,
    local int *local_his)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset;
    int i;

    i = localId;
    while (i < buckets) {
        local_his[i] = his[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = globalId;
    while (i < length) {
        offset = GET_X_VALUE(d_in, i) & mask;
        int pos = atomic_inc(local_his+offset);
        d_out[pos] = d_in[i];
#ifdef KVS_SOA
        d_out_values[pos] = d_in_values[i];
#endif
        i += globalSize;
    }
}

/*block-level split on key-value data with data reordering*/
kernel void WG_reorder_scatter(
    global const Tuple *d_in,
    global Tuple *d_out,
#ifdef KVS_SOA
    global const Tuple *d_in_values,
    global Tuple *d_out_values,
#endif
    int length,
    int buckets,
    global int *his_scanned,     /*scanned histogram*/
    local int* local_start_ptrs, /*start pos of each bucket in local mem, bucket elements*/
    global int *his_origin,      /*original histogram*/
    local Tuple *reorder_buffer  /*tuple buffer for AOS*/
#ifdef KVS_SOA
    ,local Tuple *reorder_buffer_values   /*value buffer for SOA*/
#endif
){
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    unsigned offset;
    int i;

    //load the scanned histogram
    i = localId;
    local_start_ptrs[0] = 0;
    while (i < buckets) {
        local_start_ptrs[i+1] = his_origin[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scan the local ptrs exclusively
//    LOCAL_SCAN(local_start_ptrs, buckets,1);
    scan_local(local_start_ptrs+1, buckets);

    //scatter the input to the local memory
    i = globalId;
    while (i < length) {
        offset = GET_X_VALUE(d_in, i) & mask;
        int acc = atomic_inc(local_start_ptrs+offset+1);

        /*write to the buffer*/
        reorder_buffer[acc] = d_in[i];
#ifdef KVS_SOA
        reorder_buffer_values[acc] = d_in_values[i];
#endif
        i += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = localId;
    while (i < buckets) {
        local_start_ptrs[i] = his_scanned[i*groupNum+groupId] - local_start_ptrs[i];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the data from the local mem to global mem (coalesced)
    i = localId;
    int local_sum = local_start_ptrs[buckets];

    while (i < local_sum) {
        offset = GET_X_VALUE(reorder_buffer, i) & mask;
        d_out[i+local_start_ptrs[offset]] = reorder_buffer[i];
#ifdef KVS_SOA
        d_out_values[i+local_start_ptrs[offset]] = reorder_buffer_values[i];
#endif
        i += localSize;
    }
}

/*---------------------- single-threaded kernels for CPUs and MICs --------------------------*/
/*
 * All the kernels are invoked with local_size=1
 * Only support KO and AOS
 */
kernel void single_histogram(
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

kernel void single_scatter(
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

#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE (64)     /*in bytes*/
#endif

#ifndef ELE_PER_CACHELINE
#define ELE_PER_CACHELINE   (CACHELINE_SIZE/sizeof(Tuple))
#endif

kernel void single_reorder_scatter(
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
    Tuple *local_buffer;
    local_buffer = (Tuple*)(reorder_buffer_all+ group_id*ELE_PER_CACHELINE*buckets);

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