#include "params.h"
#define NUM_OF_BANKS 32
 #define CON_OFFSET(n)  ((n)/NUM_OF_BANKS)
//#define CON_OFFSET(n)  (0)

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

void compute_mixed_access(
        unsigned step, unsigned global_id, unsigned global_size, unsigned len_total,
        unsigned *begin, unsigned *end)
{
    int step_log = findLog2(step);
    int tile = (len_total + global_size - 1) / global_size;

    int warp_id = global_id >> step_log;
    *begin = warp_id * step * tile + (global_id & (step-1));
    *end = (warp_id + 1) * step * tile;
    if ((*end) > len_total)    *end = len_total;
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
void local_warp_scan(local int* lo, int len_total, local int *sum) {

    const int local_id = get_local_id(0);                   //should have sign
    if (local_id >= len_total)   return;

    const unsigned local_size = get_local_size(0);
    const char warp_id = local_id >> WARP_BITS;             //warp ID
    const unsigned num_warps = local_size >> WARP_BITS;     //# warps
    const int lane = local_id & MASK;                       //should have sign
    int temp;

#define MAX_SUM_SIZE    (32)
    local int sums[MAX_SUM_SIZE];       //local temporary sums

    //1. Local warp-wise scan
    temp = lo[local_id];
//    barrier(CLK_LOCAL_MEM_FENCE);       /*need a barrier here*/

    if (lane >= 1) lo[local_id] += lo[local_id - 1];
    if (lane >= 2) lo[local_id] += lo[local_id - 2];
    if (lane >= 4) lo[local_id] += lo[local_id - 4];
    if (lane >= 8) lo[local_id] += lo[local_id - 8];
    if (lane >= 16) lo[local_id] += lo[local_id - 16];

    if (lane == WARP_SIZE - 1) sums[warp_id] = lo[local_id];   //get the warp sum
    lo[local_id] -= temp;                                        //exclusive minus
    barrier(CLK_LOCAL_MEM_FENCE);

    //2. Scan the intermediate sums
    if (warp_id == 0 && (local_id < num_warps)) {
        temp = sums[local_id];
        if (lane >= 1)      sums[local_id] += sums[local_id- 1];
        if (lane >= 2)      sums[local_id] += sums[local_id-2];
        if (lane >= 4)      sums[local_id] += sums[local_id-4];
        if (lane >= 8)      sums[local_id] += sums[local_id-8];
        if (lane >= 16)     sums[local_id] += sums[local_id-16];
        if ( (sum != NULL) && (lane == 0))  *sum = sums[num_warps-1];
        sums[local_id] -= temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //3. Add back
    lo[local_id] += sums[warp_id];
    barrier(CLK_LOCAL_MEM_FENCE);
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

void global_warp_scan(global int *lo_in, global int* lo_out, local int *sum) {

    const int localId = get_local_id(0);                    //should have sign!!!
    const unsigned localSize = get_local_size(0);
    const char warpId = localId >> WARP_BITS;           //warp ID
    const unsigned warpNum = localSize >> WARP_BITS;        //# warps
    const char lane = localId & MASK;                  //should have sign!!!

    lo_out[localId] = lo_in[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

#define MAX_SUM_SIZE    (32)
    local int sums[MAX_SUM_SIZE];       //local temporary sums

    //1. Local warp-wise scan
    int temp = lo_out[localId];
    if (lane >= 1) lo_out[localId] += lo_out[localId - 1];
    if (lane >= 2) lo_out[localId] += lo_out[localId - 2];
    if (lane >= 4) lo_out[localId] += lo_out[localId - 4];
    if (lane >= 8) lo_out[localId] += lo_out[localId - 8];
    if (lane >= 16) lo_out[localId] += lo_out[localId - 16];

    if (lane == WARP_SIZE - 1) sums[warpId] = lo_out[localId];   //get the warp sum
    lo_out[localId] -= temp;                                        //exclusive minus
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
    lo_out[localId] += sums[warpId];
#undef MAX_SUM_SIZE
}

void global_warp_scan_with_fence(global int* lo_in, global int* lo_out, local int *sum) {

    const int localId = get_local_id(0);                    //should have sign!!!
    const unsigned localSize = get_local_size(0);
    const char warpId = localId >> WARP_BITS;           //warp ID
    const unsigned warpNum = localSize >> WARP_BITS;        //# warps
    const char lane = localId & MASK;                  //should have sign!!!

#define MAX_SUM_SIZE    (128)           //need at most 64 for cpu, 128 for mic
    local int sums[MAX_SUM_SIZE];       //local temporary sums

    lo_out[localId] = lo_in[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    //1. Local warp-wise scan
    int temp = lo_out[localId];
    if (lane >= 1) lo_out[localId] += lo_out[localId - 1];
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lane >= 2) lo_out[localId] += lo_out[localId - 2];
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lane >= 4) lo_out[localId] += lo_out[localId - 4];
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lane >= 8) lo_out[localId] += lo_out[localId - 8];
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (lane >= 16) lo_out[localId] += lo_out[localId - 16];
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    if (lane == WARP_SIZE - 1) sums[warpId] = lo_out[localId];   //get the warp sum
    lo_out[localId] -= temp;                                        //exclusive minus
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
    lo_out[localId] += sums[warpId];

#undef MAX_SUM_SIZE
}

void global_sklansky_scan(global int *lo_in, global int* lo_out, int length, local int *sum) {
    int localId = get_local_id(0);
    lo_out[localId] = lo_in[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    int mask_j = (length>>1) -1;
    int mask_k = 0;
    int temp = 1;

    int localTemp = lo_out[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    int length_log = findLog2(length);
    for(int i = 0; i < length_log; i++) {
        if (localId < (length>>1)) {            //only half of the threads execute
            int para_j = (localId >> i) & mask_j;
            int para_k = localId & mask_k;

            int j = temp - 1 + (temp<<1)*para_j;
            int k = para_k;
            lo_out[j+k+1] = lo_out[j] + lo_out[j+k+1];

            mask_j >>= 1;
            mask_k = (mask_k<<1)+1;
            temp <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == length-1) *sum = lo_out[localId];
    lo_out[localId] -= localTemp;
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void global_blelloch_scan(global int* lo_in, global int* lo_out, int length, local int *sum) {
    int localId = get_local_id(0);
    int offset = 1;                         //offset: the distance of the two added numbers

    lo_out[localId] = lo_in[localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce
    for(int d = length >> 1; d > 0; d >>=1) {
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            lo_out[bi] += lo_out[ai];
        }
        offset <<= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)   {
        *sum = lo_out[length-1];
        lo_out[length-1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //sweep down
    for(int d = 1; d < length; d <<= 1) {
        offset >>= 1;
        if (localId < d) {
            int ai = offset * (2 * localId + 1) -1;
            int bi = offset * (2 * localId + 2) -1;

            int t = lo_out[ai];
            lo_out[ai] = lo_out[bi];
            lo_out[bi] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
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

/*---------------------fine-grained scan testing framework---------------------*/
/*
 * mark: here is a bug in the Intel OpenCL driver:
 * the barrier function in the kernel does not work
 * */
//kernel void local_scan_wrapper_SERIAL(
//        global const int * d_in,
//        global int * d_out,
//        const int len_total,
//        local int * lo)
//{
//    int local_id = get_local_id(0);
//    int local_size = get_local_size(0);
//    local int sum;
//
//    lo[local_id] = d_in[local_id];  /*loaded to local memory*/
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    if (local_id == 0) {
//        int acc = 0;
//        for (int i = 0; i < len_total; i++) {
//            int temp = lo[i];
//            lo[i] = acc;
//            acc += temp;
//        }
//        sum = acc;
//
//    }
//    barrier(CLK_LOCAL_MEM_FENCE); /*this barrier does not function*/
//    d_out[local_id] = lo[local_id]; /*only the WIs in the same wavefront of WI0 will wait for the barrier and finally output correct numbers*/
//}

/*
 * Testing the local scan schemes on
 * len_total: lsize(512)
 * place: local memory
 * config: single WG, lsize=512
 * a work-around for the OpenCL bug
 *
 * */
kernel void local_scan_wrapper_SERIAL(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    local int sum;

    lo[local_id] = d_in[local_id];  /*loaded to local memory*/
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        int acc = 0;
        for (int i = 0; i < len_total; i++) {
            int temp = lo[i];
            d_out[i] = acc;
            acc += temp;
        }
        sum = acc;
    }
}

kernel void local_scan_wrapper_KOGGE(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    local int sum;

    if (local_id < len_total)
        lo[local_id] = d_in[local_id];  /*loaded to local memory*/
    barrier(CLK_LOCAL_MEM_FENCE);
    local_warp_scan(lo, len_total, &sum);
//    local_warp_scan_with_fence(lo, &sum);

    if (local_id < len_total)
        d_out[local_id] = lo[local_id]; /*written to global memory*/
}

kernel void local_scan_wrapper_SKLANSKY(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    local int sum;

    lo[local_id] = d_in[local_id];  /*loaded to local memory*/
    barrier(CLK_LOCAL_MEM_FENCE);
    local_sklansky_scan(lo, len_total, &sum);
    d_out[local_id] = lo[local_id]; /*written to global memory*/
}

kernel void local_scan_wrapper_BRENT(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    local int sum;

    lo[local_id] = d_in[local_id];  /*loaded to local memory*/
    barrier(CLK_LOCAL_MEM_FENCE);
//    local_blelloch_scan(lo, len_total, &sum);
    local_blelloch_scan_no_conflict(lo, len_total, &sum);
    d_out[local_id] = lo[local_id]; /*written to global memory*/
}

kernel void global_scan_wrapper_SERIAL(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    int local_id = get_local_id(0);

    if (local_id == 0) {
        int acc = 0;
        for (int i = 0; i < len_total; i++) {
            d_out[i] = acc;
            acc += d_in[i];
        }
    }
}

kernel void global_scan_wrapper_KOGGE(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    local int sum;
    global_warp_scan_with_fence(d_in, d_out, &sum);
//    global_warp_scan(d_in, d_out, &sum);  /*on global mem, warp scan without fence is not available*/
}

kernel void global_scan_wrapper_SKLANSKY(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    local int sum;
    global_sklansky_scan(d_in, d_out, len_total, &sum);
}

kernel void global_scan_wrapper_BRENT(
        global const int * d_in,
        global int * d_out,
        const int len_total,
        local int * lo)
{
    local int sum;
    global_blelloch_scan(d_in, d_out, len_total, &sum);
}

#ifndef TILE_SIZE
#define TILE_SIZE (1)
#endif

/*matrix scan, data stored in the local memory*/
kernel void matrix_scan_lm(
        global int *d_in,
        global int *d_out,
        local int *ldata,       /*size: local_size*TILE_SIZE*sizeof(int)*/
        local int *lsum)        /*size: local_size*sizeof(int)*/
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    uint len_total = TILE_SIZE * local_size;

    uint begin, end, step = WARP_SIZE;
    compute_mixed_access(
            step, local_id, local_size, len_total,
            &begin, &end);

    for(int i = begin; i < end; i += step)
        ldata[i] = d_in[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*RSS scheme*/
    /*each WI processes tile_size elements*/
    int acc = 0;
    begin = local_id * TILE_SIZE;
    end = (local_id+1) * TILE_SIZE;
    for(int i = begin; i < end; i++)
        acc += ldata[i];
    lsum[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*local scan scheme*/
    if (local_id == 0) {
        acc = 0;
        for (int i = 0; i < local_size; i++) {
            int temp = lsum[i];
            lsum[i] = acc;
            acc += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    local int sum;
//    local_warp_scan_with_fence(lsum, &sum);

    /*final scan*/
    acc = 0;
    for(int i = begin; i < end; i++) {
        d_out[i] = acc + lsum[local_id];
        acc += ldata[i];
    }
}

/*matrix scan, data stored in the registers*/
kernel void matrix_scan_reg(
        global int *d_in,
        global int *d_out,
        local int *lsum)        /*size: local_size*sizeof(int)*/
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    uint len_total = TILE_SIZE * local_size;

    int reg[TILE_SIZE];

    /*RSS scheme*/
    /*each WI processes tile_size elements*/
    int acc = 0;
    uint offset = local_id * TILE_SIZE;
    for(int i = 0; i < TILE_SIZE; i++) {
        reg[i] = d_in[offset+i];
        acc += reg[i];
    }
    lsum[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*local scan scheme*/
    if (local_id == 0) {
        acc = 0;
        for (int i = 0; i < local_size; i++) {
            int temp = lsum[i];
            lsum[i] = acc;
            acc += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    local int sum;
//    local_warp_scan_with_fence(lsum, &sum);

    /*final scan*/
    acc = lsum[local_id];
    for(int i = 0; i < TILE_SIZE; i++) {
        d_out[offset+i] = acc;
        acc += reg[i];
    }
}

/*matrix scan, data stored in the registers and transferred via local memory*/
kernel void matrix_scan_lm_reg(
        global int *d_in,
        global int *d_out,
        local int *ldata,       /*size: local_size*TILE_SIZE*sizeof(int)*/
        local int *lsum)        /*size: local_size*sizeof(int)*/
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    uint len_total = TILE_SIZE * local_size;

    int reg[TILE_SIZE];

    /*data transferred to local memory*/
    uint begin, end, step = WARP_SIZE;
    compute_mixed_access(
            step, local_id, local_size, len_total,
            &begin, &end);

    for(int i = begin; i < end; i += step)
        ldata[i] = d_in[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*RSS scheme*/
    /*data transferred to registers*/
    int acc = 0;
    uint offset = local_id * TILE_SIZE;
    for(int i = 0; i < TILE_SIZE; i++) {
        reg[i] = ldata[offset+i];
        acc += reg[i];
    }
    lsum[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*local scan scheme*/
    if (local_id == 0) {
        acc = 0;
        for (int i = 0; i < local_size; i++) {
            int temp = lsum[i];
            lsum[i] = acc;
            acc += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    local int sum;
//    local_warp_scan_with_fence(lsum, &sum);

    /*final scan in the registers*/
    acc = lsum[local_id];
    for(int i = 0; i < TILE_SIZE; i++) {
        ldata[offset+i] = acc;
        acc += reg[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*copy to global memory*/
    for(int i = begin; i < end; i += step)
        d_out[i] = ldata[i];
}

kernel void matrix_scan_lm_serial(
        global int *d_in,
        global int *d_out,
        local int *ldata)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    uint len_total = TILE_SIZE * local_size;

    /*data transferred to local memory*/
    uint begin, end, step = WARP_SIZE;
    compute_mixed_access(
            step, local_id, local_size, len_total,
            &begin, &end);

    for(int i = begin; i < end; i += step)
        ldata[i] = d_in[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        int acc = 0;
        for (int i = 0; i < len_total; i++) {
            int temp = ldata[i];
            d_out[i] = acc;
            acc += temp;
        }
    }
}