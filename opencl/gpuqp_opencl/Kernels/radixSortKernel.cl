#include "dataDef.h"

kernel void radix_reduce(   
    global int *d_source_values,
    const int total,        //total element length
    global int* histogram,        //size: globalSize * SORT_RADIX
    const int shiftBits,
    local int *hist)
{

    int localId = get_local_id(0);
    int blockId = get_group_id(0);
    int blockSize = get_local_size(0);
    int gridSize = get_global_size(0) / get_local_size(0);

    int begin = blockId * (REDUCE_ELE_PER_THREAD * REDUCE_BLOCK_SIZE);
    int end = (blockId+1) * (REDUCE_ELE_PER_THREAD * REDUCE_BLOCK_SIZE) >= total ? total : (blockId+1) * (REDUCE_ELE_PER_THREAD * REDUCE_BLOCK_SIZE);
    int mask = SORT_RADIX - 1;

    //initialization: temp size is blockSize * SORT_RADIX
    for(int i = 0; i < SORT_RADIX; i++) {
        hist[i * blockSize + localId ] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = begin + localId; i < end; i+= blockSize) {
        int current = d_source_values[i];
        current = (current >> shiftBits) & mask;
        hist[current * blockSize + localId] ++;
    }    
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce
    const uint ratio = blockSize / SORT_RADIX;
    const uint digit = localId / ratio;
    const uint c = localId & ( ratio - 1 );

    uint sum = 0;
    for(int i = 0; i < SORT_RADIX; i++)  sum += hist[digit * blockSize + i * ratio + c];
    barrier(CLK_LOCAL_MEM_FENCE);


    hist[digit * blockSize + c] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

// #pragma unroll
    for(uint scale = ratio / 2; scale >= 1; scale >>= 1) {
        if ( c < scale ) {
            sum += hist[digit * blockSize + c + scale];
            hist[digit * blockSize + c] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //memory write
    if (localId < SORT_RADIX)    histogram[localId * gridSize + blockId] = hist[localId * blockSize];
}


kernel void radix_scatter(
#ifdef RECORDS
    global int *d_source_keys, global int *d_dest_keys, 
#endif
    global int *d_source_values, global int *d_dest_values,
    int total,
    int tileNum,                //number of tiles (blocks in reduce)
    global int *histogram,
    const int shiftBits,
    local ScatterData* sharedInfo   //shread memory recording intermediate info
#ifdef RECORDS
    ,bool isRecord
#endif
    )
{
    int localId = get_local_id(0);
    int blockId = get_group_id(0);

    const int lid_in_tile = localId & (SCATTER_TILE_THREAD_NUM - 1);
    const int tile_in_block = localId / SCATTER_TILE_THREAD_NUM;
    const int my_tile_id = blockId * SCATTER_TILES_PER_BLOCK + tile_in_block;   //"my" means for the threads in one tile.

    uint offset = 0;

    /*each threads with lid_in_tile has an offset recording the first place to write the  
     *element with digit "lid_in_tile" (lid_in_tile < SORT_RADIX)
     *
     * with lid_in_tile >= SORT_RADIX, their offset is always 0, no use
     */
    if (lid_in_tile < SORT_RADIX)    {
        offset = histogram[lid_in_tile * tileNum + my_tile_id];
    }

    int start = my_tile_id * (REDUCE_ELE_PER_THREAD * REDUCE_BLOCK_SIZE);
    int stop = start + (REDUCE_ELE_PER_THREAD * REDUCE_BLOCK_SIZE);
    int end = stop > total? total : stop;

    if (start >= end)   return;

    //each thread should run all the loops, even have reached the end
    //each iteration is called a TILE.
    for(; start < end; start += SCATTER_ELE_PER_TILE) {
        //each thread processes SCATTER_ELE_PER_THREAD consecutive keys
        //local counts for each thread:
        //recording how many same keys has been visited till now by this thread.
        unsigned char num_of_former_same_keys[SCATTER_ELE_PER_THREAD];

        //address in the localCount for each of the SCATTER_ELE_PER_THREAD element 
        unsigned char address_ele_per_thread[SCATTER_ELE_PER_THREAD];

        //put the global keys of this TILE to the shared memory, coalesced access
        for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
            const uint lo_id = lid_in_tile + i * SCATTER_TILE_THREAD_NUM;
            const int addr = start + lo_id;
            if (addr >= end)    break;                                     //important to have it to deal with numbers not regular
#ifdef RECORDS
            if (isRecord) {
                const int current_key = (addr < end)? d_source_keys[addr] : 0;
                sharedInfo[tile_in_block].keys[lo_id] = current_key;
            }
#endif
            const int current_value = (addr < end)? d_source_values[addr] : 0;
            sharedInfo[tile_in_block].values[lo_id] = current_value;
            
            sharedInfo[tile_in_block].digits[lo_id] = ( current_value >> shiftBits ) & (SORT_RADIX - 1);
        }

        //the SCATTER_ELE_PER_TILE threads will cooperate
        //How to cooperate?
        //Each threads read their own consecutive part, check how many same keys
        
        //initiate the localHis array
        for(uint i = 0; i < SORT_RADIX; i++) sharedInfo[tile_in_block].localHis[i * SCATTER_TILE_THREAD_NUM + lid_in_tile] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        //doing the per-TILE histogram counting
        for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
            //PAY ATTENTION: Here the shared memory access pattern has changed!!!!!!!
            //instead for coalesced access, here each thread processes consecutive area of 
            //SCATTER_ELE_PER_THREAD elements
            const uint lo_id = lid_in_tile * SCATTER_ELE_PER_THREAD + i;
            if (start + lo_id >= end)    break;                                     //important to have it to deal with numbers not regular

            const unsigned char digit = sharedInfo[tile_in_block].digits[lo_id];
            address_ele_per_thread[i] = digit * SCATTER_TILE_THREAD_NUM + lid_in_tile;
            num_of_former_same_keys[i] = sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]];
            sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]] = num_of_former_same_keys[i] + 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //now what have been saved?
        //1. keys: the keys for this TILE
        //2. digits: the digits for this TILE
        //3. address_ele_per_thread: the address in localCount for each element visited by a thread
        //4. num_of_former_same_keys: # of same keys before this key
        //5. localHis: storing the key counts

        //localHist structure:
        //[SCATTER_TILE_THREAD_NUM for Radix 0][SCATTER_TILE_THREAD_NUM for Radix 1]...

        //now exclusive scan the localHist:
//doing the naive scan:--------------------------------------------------------------------------------------------------------------------------------
        int digitCount = 0;

        if (lid_in_tile < SORT_RADIX) {
            uint localBegin = lid_in_tile * SCATTER_TILE_THREAD_NUM;
            unsigned char prev = sharedInfo[tile_in_block].localHis[localBegin];
            unsigned char now = 0;
            sharedInfo[tile_in_block].localHis[localBegin] = 0;
            for(int i = localBegin + 1; i < localBegin + SCATTER_TILE_THREAD_NUM; i++) {
                now = sharedInfo[tile_in_block].localHis[i];
                sharedInfo[tile_in_block].localHis[i] = sharedInfo[tile_in_block].localHis[i-1] + prev;
                prev = now;
                if (i == localBegin + SCATTER_TILE_THREAD_NUM - 1)  sharedInfo[tile_in_block].countArr[lid_in_tile] = sharedInfo[tile_in_block].localHis[i] + prev;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid_in_tile < SORT_RADIX)    digitCount = sharedInfo[tile_in_block].countArr[lid_in_tile];

        if (lid_in_tile == 0) {
            //exclusive scan for the countArr
            unsigned char prev = sharedInfo[tile_in_block].countArr[0];
            unsigned char now = 0;
            sharedInfo[tile_in_block].countArr[0] = 0;
            for(uint i = 1; i < SORT_RADIX; i++) {
                now = sharedInfo[tile_in_block].countArr[i];
                sharedInfo[tile_in_block].countArr[i] = sharedInfo[tile_in_block].countArr[i-1] + prev;
                prev = now;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if ( lid_in_tile < SORT_RADIX) {
            //scan add back
            uint localBegin = lid_in_tile * SCATTER_TILE_THREAD_NUM;
            for(uint i = localBegin; i < localBegin + SCATTER_TILE_THREAD_NUM; i++)
                sharedInfo[tile_in_block].localHis[i] += sharedInfo[tile_in_block].countArr[lid_in_tile];

            //now consider the offsets:
            //lid_in_tile which is < SORT_RADIX stores the global offset for this digit in this tile
            //here: updating the global offset
            //PAY ATTENTION: Why offset needs to deduct countArr? See the explaination in the final scatter!!
            sharedInfo[tile_in_block].bias[lid_in_tile] = offset - sharedInfo[tile_in_block].countArr[lid_in_tile];
            offset += digitCount;

        }

//end of naive scan:-------------------------------------------------------------------------------------------------------------------------------------

        //still consecutive access!!
        for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
            const unsigned char lo_id = lid_in_tile * SCATTER_ELE_PER_THREAD + i;
            if (start + lo_id >= end)    break;                                     //important to have it to deal with numbers not regular

            //position of this element(with id: lo_id) being scattered to
            uint pos = num_of_former_same_keys[i] + sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]];

            //since this access pattern is different from the scatter pattern(coalesced access), the position should be stored
            //also because this lo_id is not tractable in the scatter, thus using pos as the index instead of lo_id!!
            // both pos and lo_id are in the range of [0, SCATTER_ELE_PER_TILE)
            sharedInfo[tile_in_block].shuffle[pos] = lo_id;  
            // printf("write to shuffle[%d],value:%d\n",pos, lo_id);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //scatter back to the global memory, iterating in the shuffle array
        for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
            const uint lo_id = lid_in_tile + i * SCATTER_TILE_THREAD_NUM;   //coalesced access
            if ((int)lo_id < (int)end - (int)start) {                       //in case that some threads have been larger than the total length causing index overflow
                const unsigned char position = sharedInfo[tile_in_block].shuffle[lo_id];    //position is the lo_id above
                const unsigned char myDigit = sharedInfo[tile_in_block].digits[position];   //when storing digits, the storing pattern is lid_in_tile + i * SCATTER_TILE_THREAD_NUM, 
                //this is a bit complecated:
                //think about what we have now:
                //bias is the starting point for a cetain digit to be written to.
                //in the shuffle array, we have known that where each element should go
                //now we are iterating in the shuffle array
                //the array should be like this:
                // p0,p1,p2,p3......
                //p0->0, p1->0, p2->0, p3->1......
                //replacing the p0,p1...with the digit of the element they are pointing to, we can get 000000001111111122222....
                //so actually this for loop is iterating the 0000000111111122222.....!!!! for i=0, we deal with 0000.., for i = 1, we deal with 000111...
                
                //but pay attention:
                //for example: if we have 6 0's, 7 1's. Now for the first 1, lo_id = 6. Then addr would be wrong because we should write 
                //to bias[1] + 0 instead of bias[1] + 6. So we need to deduct the number of 0's, which is why previously bias need to be deducted!!!!!
                const uint addr = lo_id + sharedInfo[tile_in_block].bias[myDigit];
#ifdef RECORDS
                if (isRecord) {
                    d_dest_keys[addr] = sharedInfo[tile_in_block].keys[position];
                }
#endif
                d_dest_values[addr] = sharedInfo[tile_in_block].values[position];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
/*******************************   end of fast radix sort ***************************************/


//old implementations
// kernel void countHis(const global Record* source,
//                      const uint length,
//                      global uint* histogram,        //size: globalSize * SORT_RADIX
//                      local ushort* temp,           //each group has temp size of BLOCKSIZE * SORT_RADIX
//                      const uint shiftBits)
// {
//     int localId = get_local_id(0);
//     int globalId = get_global_id(0);
//     int globalSize = get_global_size(0);
    
//     int elePerThread = ceil(1.0*length / globalSize);
//     int offset = localId * SORT_RADIX;
//     uint mask = SORT_RADIX - 1;
    
//     //initialization
//     for(int i = 0; i < SORT_RADIX; i++) {
//         temp[i + offset] = 0;
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
    
//     for(int i = 0; i < elePerThread; i++) {
//         int id = globalId * elePerThread + i;
//         if (id >= length)   break;
//         int current = source[id].y;
//         current = (current >> shiftBits) & mask;
//         temp[offset + current]++;
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
    
//     for(int i = 0; i < SORT_RADIX; i++) {
//         histogram[i*globalSize + globalId] = (uint)temp[offset+i];
//     }
// }

// kernel void writeHis(const global Record* source,
//                      const uint length,
//                      const global uint* histogram,
//                      global uint* loc,              //size equal to the size of source
//                      local uint* temp,
//                      const uint shiftBits)               //each group has temp size of BLOCKSIZE * SORT_RADIX
// {
//     int localId = get_local_id(0);
//     int globalId = get_global_id(0);
//     int globalSize = get_global_size(0);
    
//     int elePerThread = ceil(1.0 *length / globalSize);     // length for each thread to proceed
//     int offset = localId * SORT_RADIX;
//     uint mask = SORT_RADIX - 1;
    
//     for(int i = 0; i < SORT_RADIX; i++) {
//         temp[offset + i] = histogram[i*globalSize + globalId];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
    
//     for(int i = 0; i < elePerThread; i++) {
//         int id = globalId * elePerThread + i;
//         if (id >= length)   break;
//         int current = source[globalId * elePerThread + i].y;
//         current = (current >> shiftBits) & mask;
//         loc[globalId * elePerThread + i] = temp[offset + current];
//         temp[offset + current]++;
//     }
//