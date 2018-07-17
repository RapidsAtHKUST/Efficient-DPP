#ifndef __PARAMS_H__
#define __PARAMS_H__

#define WARP_BITS                   (4)
#define WARP_SIZE                   (1<<(WARP_BITS))

//scan definition
#define SCAN_BITS        			(WARP_BITS)
#define SCAN_WARPSIZE               (1<<SCAN_BITS)
#define SCAN_MASK        			(SCAN_WARPSIZE-1)
#define SCAN_ELE_PER_THREAD  		(2)

#define REDUCE_ELE_PER_THREAD       (32)
#define REDUCE_BLOCK_SIZE           (128)

#define SCATTER_ELE_PER_THREAD      (8)
#define SCATTER_TILE_THREAD_NUM     (16)          //SCATTER_TILE_THREAD_NUM threads cooperate in a tile

#define SPLIT_VALUE_DEFAULT         (1024)       /*default value*/

#define EXPERIMENT_TIMES            (50)

#endif