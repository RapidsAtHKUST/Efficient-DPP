#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

#define WARP_BITS   (5)
#define WARP_SIZE   (1<<WARP_BITS)

//currently support WARP_SIZE * WARP_SIZE buckets
inline void work_group_scan(local int *arr1, local int *arr2, const int num, local int *sum) {
    int localId = get_local_id(0);

    if (num <= WARP_SIZE) {     //one level
        if (localId < num) {
            int temp = arr1[localId];
            if (localId >= 1) arr1[localId] += arr1[localId - 1];
            if (localId >= 2) arr1[localId] += arr1[localId - 2];
            if (localId >= 4) arr1[localId] += arr1[localId - 4];
            if (localId >= 8) arr1[localId] += arr1[localId - 8];
            if (localId >= 16) arr1[localId] += arr1[localId - 16];

            if (localId == 0) *sum = arr1[num-1];   //get the warp sum
            arr1[localId] -= temp;             //exclusive minus
            arr2[localId] = arr1[localId];
        }
    }
    else {          //two levels
        int lane = localId & (WARP_SIZE-1);
        int localSize = get_local_size(0);
        int warpId = localId >> WARP_BITS;
        int warpNum = localSize >> WARP_BITS;

        local int tempSums[WARP_SIZE];
        if (localId < num) {
            int temp = arr1[localId];
            if (lane >= 1) arr1[localId] += arr1[localId - 1];
            if (lane >= 2) arr1[localId] += arr1[localId - 2];
            if (lane >= 4) arr1[localId] += arr1[localId - 4];
            if (lane >= 8) arr1[localId] += arr1[localId - 8];
            if (lane >= 16) arr1[localId] += arr1[localId - 16];

            if (lane == 0) tempSums[warpId] = arr1[(warpId+1)*WARP_SIZE-1];   //get the warp sum
            arr1[localId] -= temp;             //exclusive minus
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //scan the tempSums array, only buckets/WARP_SIZE elements, very small
        if (warpId == 0) {
            int temp = tempSums[localId];
            if (lane >= 1) tempSums[localId] += tempSums[localId - 1];
            if (lane >= 2) tempSums[localId] += tempSums[localId - 2];
            if (lane >= 4) tempSums[localId] += tempSums[localId - 4];
            if (lane >= 8) tempSums[localId] += tempSums[localId - 8];
            if (lane >= 16) tempSums[localId] += tempSums[localId - 16];
            if (lane == 0) *sum = tempSums[num/WARP_SIZE-1];
            tempSums[localId] -= temp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //add back
        if (localId < num) {
            arr1[localId] += tempSums[warpId];
            arr2[localId] = arr1[localId];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

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
        local int* local_start_ptrs,    //start pos of each bucket in local mem, bucket elements
        int buckets,                    //number of radix bits
        global int *his,                //histogram not scanned
        local int* ptrs_diff,           //start position difference between global and local buckets
        local int *reorder_buffer_keys) //store the reordered elements
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    local int local_num;                                //total number of local elements
    unsigned mask = buckets - 1;
    int i;

    //load the scanned histogram
    i = localId;
    while (i < buckets) {
        local_start_ptrs[i] = his[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scan the local ptrs exclusively
    work_group_scan(local_start_ptrs, ptrs_diff, buckets, &local_num);

    //recording the difference
    i = localId;
    while (i < buckets) {
        ptrs_diff[i] = his_scanned[i*groupNum+groupId] - ptrs_diff[i];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scatter the input to the local memory
    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;
        int acc = atomic_inc(local_start_ptrs+offset);

        //write to the buffer, needs 5.6ms
        reorder_buffer_keys[acc] = d_in_keys[i];
        i += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the data from the local mem to global mem (coalesced)
    i = localId;
    while (i < local_num) {
        int offset = reorder_buffer_keys[i] & mask;
        d_out_keys[i+ptrs_diff[offset]] = reorder_buffer_keys[i];
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
        local int* ptrs_diff,           //start position difference between global and local buckets
        local int *reorder_buffer_keys,     //store the reordered keys
        local int *reorder_buffer_values)   //store the reordered values
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    local int local_num;                                //total number of local elements
    unsigned mask = buckets - 1;
    int i;

    //load the scanned histogram
    i = localId;
    while (i < buckets) {
        local_start_ptrs[i] = his[i*groupNum+groupId];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scan the local ptrs exclusively
    work_group_scan(local_start_ptrs, ptrs_diff, buckets, &local_num);

    //recording the difference
    i = localId;
    while (i < buckets) {
        ptrs_diff[i] = his_scanned[i*groupNum+groupId] - ptrs_diff[i];
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //scatter the input to the local memory
    i = globalId;
    while (i < length) {
        int offset = d_in_keys[i] & mask;                   //bucket: offset
        int acc = atomic_inc(local_start_ptrs+offset);        //1.8ms

        //write to the buffer, needs 5.6ms
        reorder_buffer_keys[acc] = d_in_keys[i];
        reorder_buffer_values[acc] = d_in_values[i];
        i += globalSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write the data from the local mem to global mem (coalesced)
    i = localId;
    while (i < local_num) {
        int offset = reorder_buffer_keys[i] & mask;
        d_out_keys[i+ptrs_diff[offset]] = reorder_buffer_keys[i];
        d_out_values[i+ptrs_diff[offset]] = reorder_buffer_values[i];
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