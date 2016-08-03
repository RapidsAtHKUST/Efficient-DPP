#ifndef INLJ_KERNEL_CL
#define INLJ_KERNEL_CL

#include "dataDefinition.h"

int bisearch(global Record *input, int num, int obj) {          //input should add "global"
    int res = -1;
    int begin = 0, end = num - 1, mid;
    
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

int CSS_bisearchInNode(local int *input, int begin, int end, int obj) {
    int res = -1;
    int mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        if (obj > input[mid])   begin = mid+1;
        else {
            res = mid;
            end = mid-1;
        }
    }
    return res;
}

int CSS_bisearch(global Record *input, int begin, int end, int obj) {
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

int searchInCSSTree(int obj, global Record *a, local int *b, int num, int m, int numOfInternalNodes, int mark)
{
    int d = 0;
    while (d <= numOfInternalNodes - 1) {
        int first = d * m;
        int last = d * m + m - 1;
        int nextBr = CSS_bisearchInNode(b, first, last, obj);
        if (nextBr != -1)
            d = d * (m + 1) + 1 + nextBr - first;
        else
            d = d * (m + 1) + m + 1;
    }
    
    int diff = d * m  - mark;
    
    return diff;
}

//d_R is sorted
kernel void countMatch(global Record *d_R,
                       int rLen,
                       global Record *d_S,
                       int sLen,
                       global uint *count,
                       global int *CSS_R,
                       local int *temp,               //size : CSS_R_size
                       int mPart,
                       int numOfInternalNodes,
                       int mark)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);
    
    int localCount = 0;
    int CSS_R_size = mPart * numOfInternalNodes;
    
    for(int pos = localId; pos < CSS_R_size; pos += localSize) {
        temp[pos] = CSS_R[pos];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = globalId; pos < sLen; pos += globalSize) {
        int s_y = d_S[pos].y;
        int diff = searchInCSSTree(s_y,d_R,temp,rLen,mPart,numOfInternalNodes,mark);
        int res = -1;
        
        if (diff < 0) {
            res = CSS_bisearch(d_R, rLen + diff, rLen + diff + mPart - 1, s_y);
        }
        else {
            res = CSS_bisearch(d_R, diff, diff + mPart - 1, s_y);
        }
        if (res != -1)  {
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
                       global const int *CSS_R,
                       local int *temp,               //size : CSS_R_size
                       int mPart,
                       int numOfInternalNodes,
                       int mark)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int globalId = get_global_id(0);
    int groupId = get_group_id(0);
    int globalSize = get_global_size(0);
    int groupNum = get_num_groups(0);
    
    int loc = count[globalId];
    int CSS_R_size = numOfInternalNodes * mPart;
    
    for(int pos = localId; pos < CSS_R_size; pos += localSize) {
        temp[pos] = CSS_R[pos];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int pos = globalId; pos < sLen; pos += globalSize) {
        Record tempS = d_S[pos];
        int s_y = tempS.y;
        int diff = searchInCSSTree(s_y,d_R,temp,rLen,mPart,numOfInternalNodes,mark);
        int res = -1;
        
        if (diff < 0) {
            res = CSS_bisearch(d_R, rLen + diff, rLen + diff + mPart - 1, s_y);
        }
        else {
            res = CSS_bisearch(d_R, diff, diff + mPart - 1, s_y);
        }
        if (res != -1)  {
            while (res < rLen &&  d_R[res].y == tempS.y) {
                d_out[loc].x = d_R[res++].x;
                d_out[loc++].y = tempS.x;
            }
        }
    }
}

#endif