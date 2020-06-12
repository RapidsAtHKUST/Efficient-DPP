#ifndef SCAN_KERNEL_CL
#define	SCAN_KERNEL_CL

#include "scan_local_kernel.cl"

#ifndef REGISTERS
#define REGISTERS (1)
#endif

/*adjacent synchronization*/
inline void
adjSyn(int blockId,
       int localId,
       global volatile int *inter,
       local int* r,
       local int *s) {
    if (localId == 0) {
        int p = 0;
        if (blockId == 0)   inter[0] = (*r);
        else {
            while ((p = inter[blockId-1]) == SCAN_INTER_INVALID) {}
            inter[blockId] = p + (*r);
        }
        *s = p;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*strided load to registers, for CPU*/
kernel
void scan(global int *d_in,
          global int *d_out,
          const int length,                   //input length
          local int *lo,                     //lo: local memory
          const int num_of_groups,            //#groups needed to be scanned
          const int R,                        //elements per thread in the registers
          const int L,                        //elements per thread in the local memory
          global int *inter) {                 //for adjacent sync
    auto localId = get_local_id(0);
    auto localSize = get_local_size(0);
    auto groupId = get_group_id(0);
    auto groupSize = get_num_groups(0);
    auto warpId = localId >> WARP_BITS;        //warp ID
    auto lane = localId & MASK;          //lane ID in the warp

    int c, l_begin_local, l_end_local, r_begin_local, r_end_local, reg[REGISTERS];
    local int gs, gss;
    int tempL = (R != 0) ? L+1 : L; //how many elements a thread processes in the local memory

    /*static work-group execution*/
    for(int w = groupId; w < num_of_groups; w += groupSize) {
        int l_begin_global = localSize * (R + L) * w;
        int r_begin_global = l_begin_global + L * localSize;

        if (R != 0) {
            r_begin_local = r_begin_global + localId * R;
            r_end_local = r_begin_global +  (localId+1)* R;
            if (r_end_local > length)   r_end_local = length;

            //load from global memory directly with strided access
            int localSum = 0;
            for(int r = 0; r < r_end_local - r_begin_local; r++) {
                reg[r] = d_in[r+r_begin_local];
                localSum += reg[r];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            lo[L*localSize + localId] = localSum;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //load to local memory
        if (L != 0) {
            l_begin_local = warpId * WARP_SIZE * L;
            l_end_local = (warpId+1) * WARP_SIZE * L;
            if (l_end_local + l_begin_global > length) l_end_local = length - l_begin_global;

            c = l_begin_local + lane;
            while (c < l_end_local) {
                lo[c] = d_in[l_begin_global + c];
                c += WARP_SIZE;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        scan_local(lo, tempL, &gs);      //local memory scan
        adjSyn(w, localId, inter, &gs, &gss);   //adjacent sync

        //add back and copy the local mem to global memory
        if (L != 0) {
            c = l_begin_local + lane;
            while (c < l_end_local) {
                d_out[l_begin_global + c] = lo[c] + gss;
                c += WARP_SIZE;
            }
        }

        if (R != 0) {
            //add back and copy the registers to global memory
            int preSum = lo[L*localSize+localId] + gss;
            for(int r = 0; r < r_end_local - r_begin_local; r++) {
                d_out[r+r_begin_local] = preSum;
                preSum += reg[r];
            }
        }
    }
}

/*strided load to registers, for GPU and MIC*/
kernel
void scan_coalesced(global int * d_in,                //d_inout: i/o array
                    global int * d_out,                  //d_out: output array
                    const int length,                   //input length
                    local int * lo,                     //lo: local memory
                    const int num_of_groups,            //#groups needed to be scanned
                    const int R,                        //elements per thread in the registers
                    const int L,                        //elements per thread in the local memory
                    global int * inter) {                 //for adjacent sync
    const unsigned localId = get_local_id(0);
    const unsigned localSize = get_local_size(0);
    const unsigned groupId = get_group_id(0);
    const unsigned groupSize = get_num_groups(0);
    const unsigned warpId = localId >> WARP_BITS;       //warp ID
    const unsigned lane = localId & MASK;          //lane ID in the warp

    int c, l_begin_local, l_end_local, r_begin_local, r_end_local, reg[REGISTERS];
    local int gs, gss;
    int tempL = (R != 0) ? L+1 : L; //how many elements a thread processes in the local memory

    /*static work-group execution*/
    for (int w = groupId; w < num_of_groups; w += groupSize) {
        int l_begin_global = localSize * (R + L) * w;
        int r_begin_global = l_begin_global + L * localSize;

        //load to local memory and then to private registers
        if (R != 0) {
            r_begin_local = warpId * WARP_SIZE * R;
            r_end_local = (warpId + 1) * WARP_SIZE * R;
            if (r_end_local + r_begin_global > length) r_end_local = length - r_begin_global;

            //load to local memory with coalesced access
            c = r_begin_local + lane;
            while (c < r_end_local) {
                lo[c] = d_in[r_begin_global + c];
                c += WARP_SIZE;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            //from local memory to registers and scan at the same time
            c = localId * R;
            int localSum = 0;
            for (int r = 0; r < R; r++) {
                reg[r] = localSum;
                localSum += lo[c + r];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            lo[L * localSize + localId] = localSum;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //load to local memory
        if (L != 0) {
            l_begin_local = warpId * WARP_SIZE * L;
            l_end_local = (warpId + 1) * WARP_SIZE * L;
            if (l_end_local + l_begin_global > length) l_end_local = length - l_begin_global;

            c = l_begin_local + lane;
            while (c < l_end_local) {
                lo[c] = d_in[l_begin_global + c];
                c += WARP_SIZE;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        scan_local(lo, tempL, &gs);      //local memory scan,0.3ms
        adjSyn(w, localId, inter, &gs, &gss);   //adjacent sync

        //add back and copy the local mem to global memory
        if (L != 0) {
            c = l_begin_local + lane;
            while (c < l_end_local) {
                d_out[l_begin_global + c] = lo[c] + gss;
                c += WARP_SIZE;
            }
        }

        if (R != 0) {
            int preSum = lo[L * localSize + localId] + gss;
            barrier(CLK_LOCAL_MEM_FENCE);

            //add back and copy the registers to local memory
            c = localId * R;
            for (int r = 0; r < R; r++) lo[c + r] = reg[r] + preSum;
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //from local memory to global memory, coalesced access
            c = r_begin_local + lane;
            while (c < r_end_local) {
                d_out[r_begin_global + c] = lo[c];
                c += WARP_SIZE;
            }
        }
    }
}
#endif
