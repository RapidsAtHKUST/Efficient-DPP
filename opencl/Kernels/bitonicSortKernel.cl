#ifndef BITONICSORT_KERNEL_CL
#define BITONICSORT_KERNEL_CL

#include "dataDefinition.h"

kernel void bitonicSort (global Record* d_source,
                         const int r_len,
                         const int base,
                         const int interval)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    for(int pos = globalId; pos < r_len; pos += globalSize) {
        int compareIndex = pos^interval;
        if (compareIndex > pos) {
            Record thisRecord = d_source[pos];
            Record compareRecord = d_source[compareIndex];
            
            if ( (pos & base) == 0 && thisRecord.y > compareRecord.y) {
                d_source[pos] = compareRecord;
                d_source[compareIndex] = thisRecord;
            }
            else if ((pos & base) != 0 && thisRecord.y < compareRecord.y) {
                d_source[pos] = compareRecord;
                d_source[compareIndex] = thisRecord;
            }
        }
    }
}

#endif