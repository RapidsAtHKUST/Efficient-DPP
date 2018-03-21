#ifndef HJ_KERNEL_CL
#define HJ_KERNEL_CL

#include "DataDef_CL.h"

//--------------------- deprecated start ------------------------

int getMasked(int input, int maskBits)
{
    int mask = ( 1 << maskBits ) - 1;
    return input  & mask;
}

int bisearchLeast(global Record *input, int num, int obj, int maskBits) {
    int res = -1;
    int begin = 0, end = num - 1, mid;

    while (begin <= end) {
        mid = (begin + end)/2;
        int masked = getMasked( input[mid].y, maskBits );
        if (obj >  masked )   begin = mid+1;
        else {
            if (obj ==  masked )  res = mid;
            end = mid-1;
        }
    }
    return res;
}

int bisearchMost(global Record *input, int num, int obj, int maskBits) {
    int res = -1;
    int begin = 0, end = num - 1, mid;

    while (begin <= end) {
        mid = (begin + end)/2;
        int masked = getMasked( input[mid].y, maskBits );
        if (obj < masked)   end = mid-1;
        else {
            if (obj == masked)  res = mid;
            begin = mid+1;
        }
    }
    return res;
}

kernel void matchCount (global const Record * d_R,
                        int rLen,
                        global const Record * d_S,
                        int sLen,
                        int maskBits,
                        global int * his,
                        int localMaxNum,
                        local Record *sTemp,
                        local int* temp)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);

    local int s_begin, s_end, r_begin, r_end;

    //binary search for the corresponding partition of R and S
    if (localId == 0)           s_begin = bisearchLeast(d_S,sLen,groupId,maskBits);
    else if (localId == 1)      s_end = bisearchMost(d_S,sLen,groupId,maskBits);
    else if (localId == 2)      r_begin = bisearchLeast(d_R,rLen,groupId,maskBits);
    else if (localId == 3)      r_end = bisearchMost(d_R,rLen,groupId,maskBits);
    temp[localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (s_begin == -1 || r_begin == -1) {
        his[globalId] = temp[localId];
        return;
    }

    int pass = 1;
    int fetchNum = s_end - s_begin + 1;
    if (fetchNum > localMaxNum)     {
        pass = ceil( 1.0 * fetchNum / localMaxNum );
        fetchNum = localMaxNum;
    }

    for(int i = 0; i < pass; i++) {
        for(int pos = localId; pos < fetchNum; pos += localSize) {
            if (s_begin + i * fetchNum + pos > s_end) {
                sTemp[pos].x = -1;
                break;
            }
            sTemp[pos] = d_S[s_begin + i * fetchNum + pos];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int pos = r_begin + localId; pos <= r_end; pos += localSize) {
            for(int probe = 0; probe < fetchNum; probe ++) {
                if (sTemp[probe].x == -1)   break;
                if (d_R[pos].y == sTemp[probe].y)   temp[localId]++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    his[globalId] = temp[localId];
}

kernel void matchWrite (global const Record * d_R,
                        int rLen,
                        global const Record * d_S,
                        int sLen,
                        global Record * d_out,
                        int maskBits,
                        global int * his,
                        int localMaxNum,
                        local Record *sTemp,
                        local int* temp)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);

    local int s_begin, s_end, r_begin, r_end;

    //binary search for the corresponding partition of R and S
    if (localId == 0)           s_begin = bisearchLeast(d_S,sLen,groupId,maskBits);
    else if (localId == 1)      s_end = bisearchMost(d_S,sLen,groupId,maskBits);
    else if (localId == 2)      r_begin = bisearchLeast(d_R,rLen,groupId,maskBits);
    else if (localId == 3)      r_end = bisearchMost(d_R,rLen,groupId,maskBits);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (s_begin == -1 || r_begin == -1) {
        return;
    }

    int pass = 1;
    int fetchNum = s_end - s_begin + 1;
    if (fetchNum > localMaxNum)     {
        pass = ceil( 1.0 * fetchNum / localMaxNum );
        fetchNum = localMaxNum;
    }

    temp[localId] =  his[globalId];
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for(int i = 0; i < pass; i++) {
        for(int pos = localId; pos < fetchNum; pos += localSize) {
            if (s_begin + i * fetchNum + pos > s_end) {
                sTemp[pos].x = -1;
                break;
            }
            sTemp[pos] = d_S[s_begin + i * fetchNum + pos];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int pos = r_begin + localId; pos <= r_end; pos += localSize) {
            for(int probe = 0; probe < fetchNum; probe ++) {
                if (sTemp[probe].x == -1)   break;
                if (d_R[pos].y == sTemp[probe].y) {
                    d_out[temp[localId]].x = d_R[pos].x;
                    d_out[temp[localId]++].y = sTemp[probe].x;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
//--------------------- deprecated  end ------------------------

// # of work-groups is equal to the number of partitions (buckets) in R and S
//the start_R and start_S array are generated through 1024 work-groups
kernel void probe  (    global const Record* d_R,
                        const int r_len,
                        global const Record* d_S,
                        const int s_len,
                        global const int *start_R,
                        global const int *start_S,
                        global int *d_out,
                        local Record* d_R_local,
                        local Record* d_S_local)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);

    unsigned r_begin, r_end, s_begin, s_end;

    r_begin = start_R[groupId];
    s_begin = start_S[groupId];

    if (groupId == groupNum - 1) {
        r_end = r_len + 1;
        s_end = s_len + 1;
    }
    else {
        r_end = start_R[groupId+1];
        s_end = start_S[groupId+1];
    }

    unsigned r_par_len = r_end - r_begin;
    unsigned s_par_len = s_end - s_begin;

    //copy my partition of d_R to d_R_local
    int i = r_begin + localId;
    while (i < r_end) {
        d_R_local[i-r_begin].x = d_R[i].x;
        d_R_local[i-r_begin].y = d_R[i].y;
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //copy my partition of d_S to d_S_local
    i = s_begin + localId;
    while (i < s_end) {
        d_S_local[i-s_begin].x = d_S[i].x;
        d_S_local[i-s_begin].y = d_S[i].y;
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int res = 0;
    i = localId;

    //R as the outer relation
    while (i < r_par_len) {
        for(int j = 0; j < s_par_len; j++) {
            if (d_R_local[i].x == d_S_local[j].x) {
                res ++;
            }
        }
        i += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&d_out[0], res);
}


#endif