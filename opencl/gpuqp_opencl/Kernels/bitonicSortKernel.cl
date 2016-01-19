#ifndef BITONICSORT_KERNEL_CL
#define BITONICSORT_KERNEL_CL

#include "dataDefinition.h"

kernel void bitonicSort (global Record* reSource,
                         const int group,
                         const int length,
                         const int dir,
                         const int flip)
{
    int groupId = get_group_id(0);
    int groupNum = get_num_groups(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    
    for(int gpos = groupId; gpos < group; gpos += groupNum) {
        for(int pos = localId; pos < length/2; pos += localSize) {
            int begin = gpos * length;
            int delta;
            if (flip == 1)      delta = length - 1;
            else                delta = length/2;

            int a = begin + pos;
            int b = begin + delta - flip * pos;

            if ( dir == (reSource[a].y > reSource[b].y)) {
                Record temp = reSource[a];
                reSource[a] = reSource[b];
                reSource[b] = temp;
            }
        }
    }
}

#endif