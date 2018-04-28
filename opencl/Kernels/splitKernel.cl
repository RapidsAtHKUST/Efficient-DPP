#ifndef SPLIT_KERNEL_CL
#define SPLIT_KERNEL_CL

kernel void histogram(  global const int* d_in_keys,     //input data keys
                        int length,                     //input data length
                        global int *his,                //output to the histogram array
                        local int* local_his,           //size: buckets * sizeof(int)
                        short bucketBits)               //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    int buckets = (1<<bucketBits);
    unsigned  mask = buckets - 1;

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

kernel void scatterWithHistogram(   global const int *d_in_keys,
                                    global const int *d_in_values,
                                    global int *d_out_keys,
                                    global int *d_out_values,
                                    int length,
                                    global int *his, //output to the histogram array
                                    local int* local_his,  //size: buckets * sizeof(int)
                                    short bucketBits)      //number of radix bits
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int i;

    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    int buckets = (1<<bucketBits);
    unsigned mask = buckets - 1;

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
        d_out_values[pos] = d_in_values[i];

        i += globalSize;
    }
}

//gather the start position of each partition
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