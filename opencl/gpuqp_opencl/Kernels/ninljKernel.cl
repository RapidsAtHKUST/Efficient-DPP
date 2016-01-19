#ifndef NINLJ_KERNEL_CL
#define NINLJ_KERNEL_CL

#include "dataDefinition.h"

kernel void countMatch(global const Record *d_R,
                       int rLen,
                       global const Record *d_S,
                       int sLen,
                       global uint *count,
                       local Record *temp,
                       int tempSize,
                       local uint *localCount,              //size: BLOCKSIZE
                       int pass)                            //start place for this position
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);
    
    int startPos = pass *  groupNum * tempSize + tempSize * groupId;                      //start scanning position on table S for this thread
    
    localCount[localId] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //check for surplus block
    if (startPos >= sLen)   {
        count[globalId + pass * globalSize] = 0;
        return;
    }
    
    for(int pos = localId; pos < tempSize ; pos += localSize) {
        if (startPos + pos >= sLen) {
            temp[pos].x = temp[pos].y = -1;                                   //end
            break;
        }
        temp[pos] = d_S[startPos + pos];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = localId; pos < rLen; pos += localSize) {      //size of each partition of R is localSize(BLOCKSIZE)
        for(int i = 0; i< tempSize; i++) {
            if (temp[i].x == -1)    break;                      //face an end
            if (d_R[pos].y == temp[i].y)    localCount[localId]++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    count[globalId + pass * globalSize] = localCount[localId];
}

kernel void writeMatch(global const Record *d_R,
                       int rLen,
                       global const Record *d_S,
                       int sLen,
                       global Record *h_out,
                       global uint *count,
                       local Record *temp,
                       int tempSize,
                       local uint *localCount,                  //size: BLOCKSIZE
                       uint pass)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int groupId = get_group_id(0);
    int globalSize = get_global_size(0);
    int groupNum = get_num_groups(0);
    
    int startPos = pass *  groupNum * tempSize + tempSize * groupId;                      //start scanning position on table S for this thread
    
    //check for surplus block
    if (startPos >= sLen) return;
    
    for(int pos = localId; pos < tempSize ; pos += localSize) {
        if (startPos + pos >= sLen) {
            temp[pos].x = temp[pos].y = -1;                     //shall be done!!
            break;
        }
        temp[pos] = d_S[startPos + pos];
    }
    localCount[localId] = count[globalId + pass * globalSize];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = localId; pos < rLen; pos += localSize) {      //size of each partition of R is localSize(BLOCKSIZE)
        for(int i = 0; i< tempSize; i++) {
            if (temp[i].x == -1)    break;
            if (d_R[pos].y == temp[i].y )    {
                h_out[localCount[localId]].x = d_R[pos].x;
                h_out[localCount[localId]++].y = temp[i].x;
            }
        }
    }
}

#endif