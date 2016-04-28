#ifndef __DATA_DEF_H__
#define __DATA_DEF_H__

//scan
//warp size is 32(2^5)
#define SCAN_BITS        			(5)
#define SCAN_WARPSIZE    			(1 << SCAN_BITS)
#define SCAN_MASK        			((1<<SCAN_BITS)-1)

#define SCAN_ELEMENT_PER_THREAD 	(2)
#define SCAN_MAX_BLOCKSIZE   		(1024)      //a block can have at most 1024 threads
#define SCAN_MAX_WARP_NUM        	(32)


//radix sort
#define SORT_BITS 					(4)
#define SORT_RADIX 					(1 << SORT_BITS)

#define REDUCE_ELE_PER_THREAD       (32)
#define REDUCE_BLOCK_SIZE           (128)

/*pay attention:
 * The ScatterData data structure may use char to record number count for speeding up. If the count is >= 256, it will cause problem!
 *  So a tile should not contain too many elements, i.e, SCATTER_ELE_PER_TILE should not be too large.
 *
 */
#define SCATTER_ELE_PER_THREAD      (8)
#define SCATTER_TILE_THREAD_NUM     (16)          //SCATTER_TILE_THREAD_NUM threads cooperate in a tile
//in one loop a TILE of data is processed at the same time
//IMPORTANT: make sure that SCATTER_ELE_PER_TILE is less than sizeof(unsigned char) because the internal shared variable is uchar at most this large!
#define SCATTER_ELE_PER_TILE       	(SCATTER_ELE_PER_THREAD * SCATTER_TILE_THREAD_NUM)
#define SCATTER_BLOCK_SIZE          (64)
 //num of TILES to process for each scatter block
#define SCATTER_TILES_PER_BLOCK		(SCATTER_BLOCK_SIZE / SCATTER_TILE_THREAD_NUM)

#endif