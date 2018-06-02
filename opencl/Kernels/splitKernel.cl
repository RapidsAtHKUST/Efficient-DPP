#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

#define WARP_BITS   (5)
#define WARP_SIZE   (1<<WARP_BITS)

#ifdef SMALLER_WARP_SIZE        //num <= WARP_SIZE
    #define LOCAL_SCAN(arr,num,offset)                                      \
    if (localId < num) {                                                    \
        int temp = arr[localId+offset];                                     \
        if (localId >= 1) arr[localId+offset] += arr[localId - 1+offset];   \
        if (localId >= 2) arr[localId+offset] += arr[localId - 2+offset];   \
        if (localId >= 4) arr[localId+offset] += arr[localId - 4+offset];   \
        if (localId >= 8) arr[localId+offset] += arr[localId - 8+offset];   \
        if (localId >= 16) arr[localId+offset] += arr[localId - 16+offset]; \
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
       if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];      \
       if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];      \
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
       if (lane >= 2) tempSums[localId] += tempSums[localId - 2];           \
       if (lane >= 4) tempSums[localId] += tempSums[localId - 4];           \
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
    if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];         \
    if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];         \
    if (lane >= 8) arr[localId+offset] += arr[localId - 8 +offset];         \
    if (lane >= 16) arr[localId+offset] += arr[localId - 16 +offset];       \
    if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];   \
    arr[localId+offset] -= temp0;                                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
                                                                            \
    if (warpId == 0) {                                                      \
        int temp = tempSums[localId];                                       \
        if (lane >= 1) tempSums[localId] += tempSums[localId - 1];          \
        if (lane >= 2) tempSums[localId] += tempSums[localId - 2];          \
        if (lane >= 4) tempSums[localId] += tempSums[localId - 4];          \
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

//# buckets > localSize, each thread processes (loops) elements
// void temp_scan(local int* arr, int num, int offset, global int* help, global int* help2) {
//     const int localId = get_local_id(0);
//     const int localSize = get_local_size(0);
//     const int lane = localId & (WARP_SIZE-1);
//     int warpId = localId >> WARP_BITS;
//     int warpNum = localSize >> WARP_BITS;
//     int loops = (num + localSize - 1)/localSize;

//     local int tempSums[WARP_SIZE];
//     int myPrivate[3];    

//     //local to registers
//     for(int i = 0; i < loops; i++) myPrivate[i] = arr[localId*loops + i + offset];
//     barrier(CLK_LOCAL_MEM_FENCE);

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

// int b = get_group_id(0);
// if (b==0)   help[localId] = arr[localId];
// barrier(CLK_LOCAL_MEM_FENCE);


//     //now arr has only (localSize) elements
//     if (lane >= 1) arr[localId+offset] += arr[localId - 1 +offset];     
//     if (lane >= 2) arr[localId+offset] += arr[localId - 2 +offset];     
//     if (lane >= 4) arr[localId+offset] += arr[localId - 4 +offset];     
//     if (lane >= 8) arr[localId+offset] += arr[localId - 8 +offset];     
//     if (lane >= 16) arr[localId+offset] += arr[localId - 16 +offset];   
//     if (lane == 0) tempSums[warpId] = arr[(warpId+1)*WARP_SIZE-1+offset];
//     arr[localId+offset] -= temp0;  
//     barrier(CLK_LOCAL_MEM_FENCE);

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

//     int myLocalSum = arr[localId+offset] + tempSums[warpId];
//     barrier(CLK_LOCAL_MEM_FENCE);

//     //put registers data back to the local memory
//     for(int i = 0; i < loops; i++)
//         arr[localId*loops + i + offset] = myPrivate[i] + myLocalSum;
//     barrier(CLK_LOCAL_MEM_FENCE);
// }

//thread-level histogram: each thraed has a private histogram stored in the local memory
kernel void thread_histogram(
        global const int* d_in_keys,     //input data keys
        int length,               //input data length
        global int *his,          //output to the histogram array
        int buckets,
        local int *local_buckets)
{
    int globalId = get_global_id(0);
    int globalId_fixed = globalId;
    int globalSize = get_global_size(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    unsigned mask = buckets - 1;

    int idx = localId;
    while (idx < buckets * localSize) {
        local_buckets[idx] = 0;
        idx += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (globalId < length) {
        int offset = d_in_keys[globalId] & mask;
        local_buckets[offset*localSize+localId]++;
        globalId += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the histogram back to the global memory
    for(int i = 0; i < buckets; i++) {
        his[i*globalSize+globalId_fixed] = local_buckets[i*localSize+localId];
    }
}

kernel void thread_scatter_k(
        global const int  *d_in_keys,
        global int  *d_out_keys,
        int length,
        global int *his,         //histogram not scanned
        int buckets,
        local int *local_buckets)          //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    unsigned mask = buckets - 1;

    for(int i = 0; i < buckets; i++) {
        local_buckets[i*localSize + localId] = his[i*globalSize+globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (globalId < length) {
        int offset = d_in_keys[globalId] & mask;
        d_out_keys[local_buckets[offset*localSize + localId]++] = d_in_keys[globalId];
        globalId += globalSize;
    }
}

kernel void thread_scatter_kv(
        global const int  *d_in_keys,
        global const int  *d_in_values,
        global int  *d_out_keys,
        global int  *d_out_values,
        int length,
        global int *his,         //histogram not scanned
        int buckets,
        local int *local_buckets)          //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    unsigned mask = buckets - 1;

    for(int i = 0; i < buckets; i++) {
        local_buckets[i*localSize + localId] = his[i*globalSize+globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (globalId < length) {
        int offset = d_in_keys[globalId] & mask;
        int idx = offset*localSize + localId;
        d_out_keys[local_buckets[idx]] = d_in_keys[globalId];
        d_out_values[local_buckets[idx]++] = d_in_values[globalId];
        globalId += globalSize;
    }
}

//block-level histogram: threads in a block share a histogram
kernel void block_histogram(
    global const int* d_in_keys,     //input data keys
    int length,               //input data length
    global int *his,          //output to the histogram array
    local int* local_his,     //size: buckets * sizeof(int)
    int buckets)          //number of buckets
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;

    //local histogram initialization
    int i = localId;
    while (i < buckets) {
        local_his[i] = 0;
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;
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

kernel void block_scatter_k(
        global const int  * d_in_keys,
        global int  * d_out_keys,
        int length,
        global int *his,         //histogram not scanned
        local int* local_his,   //size: buckets * sizeof(int)
        int buckets)          //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    int i;

    i = localId;
    while (i < buckets) {
        local_his[i] = his[i*groupNum+groupId];     //scatter global array to local array
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;
        int pos = atomic_inc(local_his+offset);     //should atomic_inc and return the old value
        d_out_keys[pos] = d_in_keys[i];
        i += globalSize;
    }
}

kernel void block_scatter_kv(
        global const int  * d_in_keys,
        global const int  * d_in_values,
        global int  * d_out_keys,
        global int  * d_out_values,
        int length,
        global int *his,         //histogram not scanned
        local int* local_his,   //size: buckets * sizeof(int)
        int buckets)          //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    int i;

    i = localId;
    while (i < buckets) {
        local_his[i] = his[i*groupNum+groupId];     //scatter global array to local array
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;
        int pos = atomic_inc(local_his+offset);     //should atomic_inc and return the old value
        //~9ms for read and write
        d_out_keys[pos] = d_in_keys[i];
        d_out_values[pos] = d_in_values[i];

        i += globalSize;
    }
}

//block-level split on key-only data with data reordering
kernel void block_reorder_scatter_k(
        global const int *d_in_keys,
        global int *d_out_keys,
        int length,
        global int *his_scanned,            //histogram scanned
        local int* local_start_ptrs,    //start pos of each bucket in local mem and local sum, (bucket+1) elements
        int buckets,                    //number of radix bits
        global int *his,                //histogram not scanned
        local int *reorder_buffer_keys) //store the reordered elements
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    int i;

    //load the scanned histogram
    i = localId;
    local_start_ptrs[0] = 0;            //for the first bucket
    while (i < buckets) {
        local_start_ptrs[i+1] = his[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scan the local ptrs exclusively
    LOCAL_SCAN(local_start_ptrs, buckets,1);

    //scatter the input to the local memory
    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;

        //after looping, the ptr of bucket m will point to bucket m+1
        int acc = atomic_inc(local_start_ptrs+offset+1);
        reorder_buffer_keys[acc] = d_in_keys[i];
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
        int offset = reorder_buffer_keys[i] & mask;
        d_out_keys[i+local_start_ptrs[offset]] = reorder_buffer_keys[i];
        i += localSize;
    }
}

//block-level split on key-value data with data reordering
kernel void block_reorder_scatter_kv(
        global const int *d_in_keys,
        global const int *d_in_values,
        global int *d_out_keys,
        global int *d_out_values,
        int length,
        global int *his_scanned,            //histogram scanned
        local int* local_start_ptrs,    //start pos of each bucket in local mem, bucket elements
        int buckets,                    //number of radix bits
        global int *his,                //histogram not scanned
        local int *reorder_buffer_keys,     //store the reordered keys
        local int *reorder_buffer_values)   //store the reordered values
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned mask = buckets - 1;
    int i;

    //load the scanned histogram
    i = localId;
    local_start_ptrs[0] = 0;
    while (i < buckets) {
        local_start_ptrs[i+1] = his[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scan the local ptrs exclusively
    LOCAL_SCAN(local_start_ptrs, buckets,1);

    //scatter the input to the local memory
    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;                   //bucket: offset
        int acc = atomic_inc(local_start_ptrs+offset+1);

        //write to the buffer,
        reorder_buffer_keys[acc] = d_in_keys[i];
        reorder_buffer_values[acc] = d_in_values[i];
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
        int offset = reorder_buffer_keys[i] & mask;
        d_out_keys[i+local_start_ptrs[offset]] = reorder_buffer_keys[i];
        d_out_values[i+local_start_ptrs[offset]] = reorder_buffer_values[i];
        i += localSize;
    }
}

//gather the start position of each partition (optional)
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

#endif