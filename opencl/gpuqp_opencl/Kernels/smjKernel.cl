#ifndef SMJ_KERNEL_CL
#define SMJ_KERNEL_CL

#include "dataDefinition.h"

//binary search in the certain R block
int bisearch(global Record *input, int begin, int end, int obj) {
    int res = -1;
    int mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        if (obj > input[mid].y)   begin = mid+1;
        else {
            if (obj == input[mid].y)  res = mid;
            end = mid-1;
        }
    }
    return res;
}

int findBack(global Record *d_R, int rLen, int obj) {
    int res = rLen-1;
    bool equFound = false;          //check whether the equal one has been found
    
    int begin = 0, end = rLen - 1, mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        int temp = d_R[mid].y;
        if (obj < temp)   {
            if (!equFound)  res = mid;
            end = mid - 1;
        }
        else if (obj == temp) {
            begin = mid + 1;
            res = mid;
            equFound = true;
        }
        else {
            begin = mid + 1;
        }
    }
    if ( (!equFound) && res == 0 )  res = -1;       //not in this scale
    return res;
}

int findFore(global Record *d_R, int rLen, int obj) {
    int res = 0;
    bool equFound = false;          //check whether the equal one has been found
    
    int begin = 0, end = rLen - 1, mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        int temp = d_R[mid].y;
        if (obj < temp)   {
            end = mid - 1;
        }
        else if (obj == temp) {
            end = mid - 1;
            res = mid;
            equFound = true;
        }
        else {
            if (!equFound)  res = mid;
            begin = mid + 1;
        }
    }
    if ( (!equFound) && res == rLen-1 )  res = -1;       //not in this scale
    return res;
}

//both d_R and d_S are sorted
kernel void countMatch(global Record *d_R,
                       int rLen,
                       global Record *d_S,
                       int sLen,
                       global uint *count,
                       local Record *temp,               //size : local_S_Length
                       int local_S_length)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);
    
    int localCount = 0;
    local int beginR, endR, beginS, endS,sTempNum;
    
    if (localId == 0)   {
        beginS = groupId * local_S_length;
        endS =  (groupId + 1) * local_S_length;
        if (endS > sLen)    endS = sLen;
        sTempNum = endS - beginS;               //number of S records proceeded in this block
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = localId ; pos < sTempNum; pos += localSize ) {
        temp[pos] = d_S[pos + beginS];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localId == 0) {
        beginR = findFore(d_R,rLen,temp[0].y);
    }
    else if (localId == 1) {
        endR = findBack(d_R,rLen,temp[sTempNum-1].y);       //inclusive
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (beginR == -1 || endR == -1) {
        count[globalId] = localCount;
        return;         //no matching block
    }
    
    int curLeast = beginR;
    
    for(int pos = localId; pos < sTempNum; pos += localSize) {
        int s_y = temp[pos].y;
        int res = bisearch(d_R,curLeast,endR,s_y);
        if (res != -1)  {
            if (res > curLeast) curLeast = res;
            while (res < rLen && d_R[res].y == s_y ) {
                localCount++;
                res++;
            }
        }
    }
    count[globalId] = localCount;
}

kernel void writeMatch(global Record *d_R,
                       int rLen,
                       global Record *d_S,
                       int sLen,
                       global Record *d_out,
                       global uint *count,
                       local Record *temp,               //size : local_S_length
                       int local_S_length)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int groupId = get_group_id(0);
    int globalSize = get_global_size(0);
    int groupNum = get_num_groups(0);
    
    local int beginR, endR, beginS, endS,sTempNum;
    
    if (localId == 0)   {
        beginS = groupId * local_S_length;
        endS =  ( (groupId + 1) * local_S_length <= sLen ? ( (groupId + 1) * local_S_length ) : sLen ); //exclusive
        sTempNum = endS - beginS;               //number of S records proceeded in this block
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = localId ; pos < sTempNum; pos += localSize ) {
        temp[pos] = d_S[pos + beginS];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localId == 0) {
        beginR = findFore(d_R,rLen,temp[0].y);
    }
    else if (localId == 1) {
        endR = findBack(d_R,rLen,temp[sTempNum-1].y);       //inclusive
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (beginR == -1 || endR == -1) return;         //no matching block
    
    uint loc = count[globalId];
    int curLeast = beginR;
    
    for(int pos = localId; pos < sTempNum; pos += localSize) {
        int s_y = temp[pos].y;
        int res = bisearch(d_R,curLeast,endR,s_y);
        if (res != -1)  {
            if (res > curLeast) curLeast = res;
            while (res < rLen && d_R[res].y == s_y ) {
                d_out[loc].x = d_R[res++].x;
                d_out[loc++].y = temp[pos].x;
            }
        }
    }
}


#endif