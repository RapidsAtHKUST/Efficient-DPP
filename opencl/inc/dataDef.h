#ifndef __DATA_DEF_H__
#define __DATA_DEF_H__

//throughput processing
#define VPU_REPEAT_TIME             (120)               //repeat time for madd operation
#define FACTOR                      (1.0001)            //madd: s = s * FACTOR + FACTOR


//scan definition
#define SCAN_BITS        			(4)
#define SCAN_MASK        			((1<<SCAN_BITS)-1)
#define SCAN_ELE_PER_THREAD  		(2)
#define SCAN_WARPSIZE    			(1<<SCAN_BITS)
#define SCAN_MAX_BLOCKSIZE   		(1024)      //a block can have at most 1024 threads

//radix sort definition
#define SORT_BITS					(4)
#define SORT_RADIX					(1<<SORT_BITS)

#define REDUCE_ELE_PER_THREAD       (32)
#define REDUCE_BLOCK_SIZE           (128)

#define SCATTER_ELE_PER_THREAD      (8)
#define SCATTER_TILE_THREAD_NUM     (16)          //SCATTER_TILE_THREAD_NUM threads cooperate in a tile

//in one loop a TILE of data is processed at the same time
//IMPORTANT: make sure that SCATTER_ELE_PER_TILE is less than sizeof(unsigned char) because the internal shared variable is uchar at most this large!
#define SCATTER_ELE_PER_TILE       	(SCATTER_ELE_PER_THREAD * SCATTER_TILE_THREAD_NUM)
#define SCATTER_BLOCK_SIZE          (64)
 //num of TILES to process for each scatter block
#define SCATTER_TILES_PER_BLOCK		(SCATTER_BLOCK_SIZE / SCATTER_TILE_THREAD_NUM)

//radix sort intermediate data structure
typedef struct {
    unsigned char digits[SCATTER_ELE_PER_TILE];        //store the digits 
    unsigned char shuffle[SCATTER_ELE_PER_TILE];       //the positions that each elements in the TILE should be scattered to
    unsigned char localHis[SCATTER_TILE_THREAD_NUM * SORT_RADIX];    //store the digit counts for a TILE
    unsigned char countArr[SORT_RADIX];
    uint bias[SORT_RADIX];                           //the global offsets of the radixes in this TILE
    int values[SCATTER_ELE_PER_TILE];
#ifdef RECORDS
    int keys[SCATTER_ELE_PER_TILE];            //store the keys
#endif
} ScatterData;

//CSS tree structer for INLJ
typedef struct CSS_Tree_Info {
	cl_mem d_CSS;
	int CSS_length;
	int mPart;
	int numOfInternalNodes;
	int mark;
} CSS_Tree_Info;

//Basic test information
typedef struct Basic_info {
    //vpu testing
    double vpu_time;
    double vpu_throughput;

    //memory bandwidth
    double mem_read_time;
    double mem_read_throughput;
    double mem_write_time;
    double mem_write_throughput;
} Basic_info;

typedef struct Device_perf_info {
    Basic_info float_info;
    Basic_info double_info;
} Device_perf_info;

#endif