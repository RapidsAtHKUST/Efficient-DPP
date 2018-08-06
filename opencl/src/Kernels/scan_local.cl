#include "params.h"
#define NUM_OF_BANKS 4
// #define CON_OFFSET(n)  ((n)/NUM_OF_BANKS)
#define CON_OFFSET(n)  (0)

//log function based on look-up table
int findLog2(int input) {
    int lookup[21] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576};
    int start = 0, end = 21, middle = (start+end)/2;

    while(lookup[middle] != input) {
        if (start >= end)   return -1;
        if (input > lookup[middle])  start = middle+1;
        else                         end = middle-1;
        middle = (start+end)/2;
    }
    return middle;
}

//----------------------------- basic local scan networks --------------------------
//local serial scan
inline void local_serial_scan(local int* lo, int length, local int *sum) {
    int localId = get_local_id(0);
    if (localId == 0) {
        int sum1 = lo[0];
        lo[0] = 0;
        for(int i = 1; i < length; i++) {
            int cur = lo[i];
            lo[i] = sum1;
            sum1 += cur;
        }
        *sum = sum1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//try different number of threads used in the serial scan
inline void local_serial_scan_multithread(local int* lo, int length, local int *sum) {
    int localId = get_local_id(0);
    int thread_used = 128;
    int ele_per_thread = length/thread_used;

    local int tempSums[128];

    //1.reduce
    if (localId < thread_used) {
        int start = localId * ele_per_thread;
        int sum1 = 0;
        for(int i = 0; i < ele_per_thread; i++) {
            sum1 += lo[start+i];
        }
        tempSums[localId] = sum1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //2.scan
    if (localId == 0) {
        int local_temp0 = tempSums[0];
        tempSums[0] = 0;
        for(int r = 1; r < thread_used; r++) {
            int local_temp1 = tempSums[r];
            tempSums[r] = local_temp0 + tempSums[r-1];
            local_temp0 = local_temp1;
        }
        *sum = local_temp0 + tempSums[thread_used-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //3.scan
    if (localId < thread_used) {
        int start = localId * ele_per_thread;
        int local_temp0 = lo[start];
        lo[start] = tempSums[localId];
        for(int r = 1; r < ele_per_thread; r++) {
            int local_temp1 = lo[r+start];
            lo[r+start] = local_temp0 + lo[r+start-1];
            local_temp0 = local_temp1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//local blelloch scan (minimum size)
inline void local_blelloch_scan(local int* lo, int length, local int *sum) {
    int localId = get_local_id(0);
    int offset = 1;                         //offset: the distance of the two added numbers

    //reduce
    for(int d = length >> 1; d > 0; d >>=1) {
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            lo[bi] += lo[ai];
        }
        offset <<= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)   {
        *sum = lo[length-1];
        lo[length-1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //sweep down
    for(int d = 1; d < length; d <<= 1) {
        offset >>= 1;
        if (localId < d) {
            int ai = offset * (2 * localId + 1) -1;
            int bi = offset * (2 * localId + 2) -1;

            int t = lo[ai];
            lo[ai] = lo[bi];
            lo[bi] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//blelloch scan with no bank conflict
inline void local_blelloch_scan_no_conflict(local int* lo, int length, local int *sum) {
    int localId = get_local_id(0);
    int offset = 1;                         //offset: the distance of the two added numbers

    //reduce
    for(int d = length >> 1; d > 0; d >>=1) {
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            ai += CON_OFFSET(ai);
            bi += CON_OFFSET(bi);
            lo[bi] += lo[ai];
        }
        offset <<= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)   {
        *sum = lo[length-1 + CON_OFFSET(length-1)];
        lo[length-1 + CON_OFFSET(length-1)] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //sweep down
    for(int d = 1; d < length; d <<= 1) {
        offset >>= 1;
        if (localId < d) {
            int ai = offset * (2 * localId + 1) -1;
            int bi = offset * (2 * localId + 2) -1;
            ai += CON_OFFSET(ai);
            bi += CON_OFFSET(bi);

            int t = lo[ai];
            lo[ai] = lo[bi];
            lo[bi] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//local sklansky scan
void local_sklansky_scan(local int* lo, int length, local int *sum) {
    int localId = get_local_id(0);

    int mask_j = (length>>1) -1;
    int mask_k = 0;
    int temp = 1;

    int localTemp = lo[localId];
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

    if (localId == length-1) *sum = lo[localId];
    lo[localId] -= localTemp;
    barrier(CLK_LOCAL_MEM_FENCE);
}

//warp-wise intra-block scan (minimum depth), each thread only processes at most 1 element
void local_warp_scan(local int* lo, local int *sum) {

    const int localId = get_local_id(0);                    //should have sign!!!
    const unsigned localSize = get_local_size(0);
    const char warpId = localId >> WARP_BITS;           //warp ID
    const unsigned warpNum = localSize >> WARP_BITS;        //# warps
    const char lane = localId & MASK;                  //should have sign!!!

#define MAX_SUM_SIZE    (32)
    local int sums[MAX_SUM_SIZE];       //local temporary sums

    //1. Local warp-wise scan
    int temp = lo[localId];
    if (lane >= 1) lo[localId] += lo[localId - 1];
    if (lane >= 2) lo[localId] += lo[localId - 2];
    if (lane >= 4) lo[localId] += lo[localId - 4];
    if (lane >= 8) lo[localId] += lo[localId - 8];
    if (lane >= 16) lo[localId] += lo[localId - 16];

    if (lane == WARP_SIZE - 1) sums[warpId] = lo[localId];   //get the warp sum
    lo[localId] -= temp;                                        //exclusive minus
    barrier(CLK_LOCAL_MEM_FENCE);

    //2. Scan the intermediate sums
    if (warpId == 0 && localId < warpNum) {
        temp=sums[localId];
        if (lane >= 1)      sums[localId] += sums[localId-1];
        if (lane >= 2)      sums[localId] += sums[localId-2];
        if (lane >= 4)      sums[localId] += sums[localId-4];
        if (lane >= 8)      sums[localId] += sums[localId-8];
        if (lane >= 16)     sums[localId] += sums[localId-16];
        if (lane == WARP_SIZE-1)  *sum = sums[warpNum-1];   //get the total sum
        sums[localId] -= temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //3. Add back
    lo[localId] += sums[warpId];
#undef MAX_SUM_SIZE
}

//warp-wise intra-block scan (minimum depth) with memory fence for CPU and MIC
void local_warp_scan_with_fence(local int* lo, local int *sum) {

    const int localId = get_local_id(0);                    //should have sign!!!
    const unsigned localSize = get_local_size(0);
    const char warpId = localId >> WARP_BITS;           //warp ID
    const unsigned warpNum = localSize >> WARP_BITS;        //# warps
    const char lane = localId & MASK;                  //should have sign!!!

#define MAX_SUM_SIZE    (128)           //new at most 64 for cpu, 128 for mic
    local int sums[MAX_SUM_SIZE];       //local temporary sums

    //1. Local warp-wise scan
    int temp = lo[localId];
    if (lane >= 1) lo[localId] += lo[localId - 1];
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (lane >= 2) lo[localId] += lo[localId - 2];
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (lane >= 4) lo[localId] += lo[localId - 4];
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (lane >= 8) lo[localId] += lo[localId - 8];
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (lane >= 16) lo[localId] += lo[localId - 16];
    mem_fence(CLK_LOCAL_MEM_FENCE);

    if (lane == WARP_SIZE - 1) sums[warpId] = lo[localId];   //get the warp sum
    lo[localId] -= temp;                                        //exclusive minus
    barrier(CLK_LOCAL_MEM_FENCE);

    //2. Scan the intermediate sums
    if (localId == 0)  {
        int sum1 = sums[0];
        sums[0] = 0;
        for(int i = 1; i < warpNum; i++) {
            int cur = sums[i];
            sums[i] = sum1;
            sum1 += cur;
        }
        *sum = sum1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //3. Add back
    lo[localId] += sums[warpId];

#undef MAX_SUM_SIZE
}

//----------------------------- end of basic local scan networks --------------------------

//intra-block matrix scan
inline void scan_local(local int* lo, int ele_per_thread, local int* totalSum) {
    const unsigned localId = get_local_id(0);
    const unsigned localSize = get_local_size(0);
    int tempStore;  //to store the first 1024 elements in the original lo array
    int tempSum;
    int row_start, row_end;

    //notice: when using conflict-free blelloch scan sub-scheme, remove the if statement
    if (ele_per_thread != 1) {      //if ele_per_thread==1, skip the serial reduction
        //1. Reduction (per thread)
        row_start = localId * ele_per_thread;
        row_end = (localId + 1) * ele_per_thread;

        int local_sum = 0;
        for (int r = row_start; r < row_end; r++) {
            local_sum += lo[r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        tempStore = lo[localId + CON_OFFSET(localId)]; //for bank-conflict-free blelloch scan
        lo[localId + CON_OFFSET(localId)] = local_sum; //for bank-conflict-free blelloch scan
        barrier(CLK_LOCAL_MEM_FENCE);
    }
//----------------------- local scan schemes -----------------------------
    //2. Very fast scan on localSize elements
    local_serial_scan(lo,localSize, totalSum);
//   local_warp_scan(lo, totalSum);                 //for GPU

//    local_warp_scan_with_fence(lo, totalSum);     //for CPU and MIC
//    local_blelloch_scan_no_conflict(lo,localSize, totalSum);
//    local_sklansky_scan(lo, localSize, totalSum);
//----------------------- local scan schemes -----------------------------

    if (ele_per_thread != 1) {      //if ele_per_thread==1, skip the serial scan
        tempSum = lo[localId + CON_OFFSET(localId)];
        lo[localId + CON_OFFSET(localId)] = tempStore; //switch back the first localSize elements
        barrier(CLK_LOCAL_MEM_FENCE);

        //3. Scan
        int local_temp0 = lo[row_start];
        lo[row_start] = tempSum;
        for (int r = row_start + 1; r < row_end; r++) {
            int local_temp1 = lo[r];
            lo[r] = local_temp0 + lo[r - 1];
            local_temp0 = local_temp1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}