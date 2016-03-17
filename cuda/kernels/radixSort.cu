//
//  radixSort.cu
//  comparison_gpu
//
//  Created by Bryan on 01/26/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"
#define BITS 4
#define RADIX (1<<BITS)
using namespace std;
__global__
void countHis(const Record* source,
              const int length,
			  int* histogram,        //size: globalSize * RADIX
              const int shiftBits)
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = (length + globalSize - 1) / globalSize;
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    //initialization
    for(int i = 0; i < RADIX; i++) {
        temp[i + offset] = 0;
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[id].y;
        current = (current >> shiftBits) & mask;
        temp[offset + current]++;
    }
    __syncthreads();
    
    for(int i = 0; i < RADIX; i++) {
        histogram[i*globalSize + globalId] = temp[offset+i];
    }
}

__global__
void writeHis(const Record* source,
			  const int length,
              const int* histogram,
              int* loc,              //size equal to the size of source
              const int shiftBits)               
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = (length + globalSize - 1) / globalSize;     // length for each thread to proceed
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    for(int i = 0; i < RADIX; i++) {
        temp[offset + i] = histogram[i*globalSize + globalId];
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[globalId * elePerThread + i].y;
        current = (current >> shiftBits) & mask;
        loc[globalId * elePerThread + i] = temp[offset + current];
        temp[offset + current]++;
    }
}

__global__
void countHis_int(const int* source,
              	  const int length,
			  	  int* histogram,        //size: globalSize * RADIX
              	  const int shiftBits)
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = (length + globalSize - 1) / globalSize;
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    //initialization
    for(int i = 0; i < RADIX; i++) {
        temp[i + offset] = 0;
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[id];
        current = (current >> shiftBits) & mask;
        temp[offset + current]++;
    }
    __syncthreads();
    
    for(int i = 0; i < RADIX; i++) {
        histogram[i*globalSize + globalId] = temp[offset+i];
    }
}

__global__
void writeHis_int(const int* source,
			  const int length,
              const int* histogram,
              int* loc,              //size equal to the size of source
              const int shiftBits)               
{
	extern __shared__ int temp[];		//each group has temp size of BLOCKSIZE * RADIX

    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int globalSize = blockDim.x * gridDim.x;
    
    int elePerThread = (length + globalSize - 1) / globalSize;     // length for each thread to proceed
    int offset = localId * RADIX;
    int mask = RADIX - 1;
    
    for(int i = 0; i < RADIX; i++) {
        temp[offset + i] = histogram[i*globalSize + globalId];
    }
    __syncthreads();
    
    for(int i = 0; i < elePerThread; i++) {
        int id = globalId * elePerThread + i;
        if (id >= length)   break;
        int current = source[globalId * elePerThread + i];
        current = (current >> shiftBits) & mask;
        loc[globalId * elePerThread + i] = temp[offset + current];
        temp[offset + current]++;
    }
}




/*
 *							Fast radix sort
 *
 *			1. each block count histograms and reduce
 *			2. grid-wise exclusive scan on the histrograms
 *			3. each block does scatter according to the histograms
 *
 **/
#define REDUCE_ELE_PER_THREAD 		32
#define REDUCE_BLOCK_SIZE			128

#define SCATTER_ELE_PER_THREAD		8
#define SCATTER_TILE_THREAD_NUM		16			//SCATTER_TILE_THREAD_NUM threads cooperate in a tile
//in one loop a batch of data is processed at the same time
//IMPORTANT: make sure that SCATTER_ELE_PER_BATCH is less than sizeof(unsigned char) because the internal shared variable is uchar at most this large!
#define SCATTER_ELE_PER_BATCH		(SCATTER_ELE_PER_THREAD * SCATTER_TILE_THREAD_NUM)
#define SCATTER_BLOCK_SIZE			64
 //num of TILES to process for each scatter block
#define SCATTER_TILES_PER_BLOCK				(SCATTER_BLOCK_SIZE / SCATTER_TILE_THREAD_NUM)

__global__
void radix_reduce(	const int* d_source,
              		const int total,		//total element length
              		const int blockLen,		//len of elements each block should process
			  		int* histogram,        //size: globalSize * RADIX
              		const int shiftBits)
{
	extern __shared__ int hist[];		

    int localId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int gridSize = gridDim.x;

    int begin = blockId * blockLen;
    int end = (blockId+1) * blockLen >= total ? total : (blockId+1) * blockLen;
    int mask = RADIX - 1;

    //initialization: temp size is blockSize * RADIX
    for(int i = 0; i < RADIX; i++) {
        hist[i * blockSize + localId ] = 0;
    }
    __syncthreads();

	for(uint i = begin + localId; i < end; i+= blockSize) {
		int current = d_source[i];
		current = (current >> shiftBits) & mask;
		hist[current * blockSize + localId] ++;
	}    
    __syncthreads();

	//reduce
    const uint ratio = blockSize / RADIX;
    const uint digit = localId / ratio;
    const uint c = localId & ( ratio - 1 );

    uint sum = 0;
    for(int i = 0; i < RADIX; i++) 	sum += hist[digit * blockSize + i * ratio + c];
    __syncthreads();


	hist[digit * blockSize + c] = sum;
	__syncthreads();

#pragma unroll
	for(uint scale = ratio / 2; scale >= 1; scale >>= 1) {
		if ( c < scale ) {
			sum += hist[digit * blockSize + c + scale];
			hist[digit * blockSize + c] = sum;
		}
		__syncthreads();
	}

	//memory write
	if (localId < RADIX)	histogram[localId * gridSize + blockId] = hist[localId * blockSize];
}

//data structures for storing information for each batch in a tile
typedef struct {
	unsigned char digits[SCATTER_ELE_PER_BATCH];		//store the digits 
	unsigned char shuffle[SCATTER_ELE_PER_BATCH];		//the positions that each elements in the batch should be scattered to
	unsigned char localHis[SCATTER_TILE_THREAD_NUM * RADIX];	//store the digit counts for a batch
    unsigned char countArr[RADIX];
	uint bias[RADIX];							//the global offsets of the radixes in this batch
	int keys[SCATTER_ELE_PER_BATCH];			//store the keys
} ScatterData;

__global__
void radix_scatter( const int *d_in,
					int *d_out,
					int total,
					int tileLen,				//length for each tile(block in reduce)
					int tileNum,				//number of tiles (blocks in reduce)
					int *histogram,
					const int shiftBits)
{
 	int localId = threadIdx.x;
    int blockId = blockIdx.x;

    const int lid_in_tile = localId & (SCATTER_TILE_THREAD_NUM - 1);
    const int tile_in_block = localId / SCATTER_TILE_THREAD_NUM;
    const int my_tile_id = blockId * SCATTER_TILES_PER_BLOCK + tile_in_block;	//"my" means for the threads in one tile.

    //shared mem data
    __shared__ ScatterData sharedInfo[SCATTER_TILES_PER_BLOCK];

    uint offset = 0;

    /*each threads with lid_in_tile has an offset recording the first place to write the  
     *element with digit "lid_in_tile" (lid_in_tile < RADIX)
     *
     * with lid_in_tile >= RADIX, their offset is always 0, no use
     */
    if (lid_in_tile < RADIX)	{
        offset = histogram[lid_in_tile * tileNum + my_tile_id];
    }

    int start = my_tile_id * tileLen;
    int stop = start + tileLen;
    int end = stop > total? total : stop;

    //each thread should run all the loops, even have reached the end
    //
    //each iteration is called a batch.
    for(; start < stop; start += SCATTER_ELE_PER_BATCH) {
    	//each thread processes SCATTER_ELE_PER_THREAD consecutive keys

    	//local counts for each thread:
    	//recording how many same keys has been visited till now by this thread.
    	unsigned char num_of_former_same_keys[SCATTER_ELE_PER_THREAD];

    	//address in the localCount for each of the SCATTER_ELE_PER_THREAD element 
    	unsigned char address_ele_per_thread[SCATTER_ELE_PER_THREAD];

    	//put the global keys of this batch to the shared memory, coalesced access
    	for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
    		const uint lo_id = lid_in_tile + i * SCATTER_TILE_THREAD_NUM;
    		const uint addr = start + lo_id;
    		const int current_key = (addr < end)? d_in[addr] : 0;
    		sharedInfo[tile_in_block].keys[lo_id] = current_key;
    		sharedInfo[tile_in_block].digits[lo_id] = ( current_key >> shiftBits ) & (RADIX - 1);
    	}

    	//the SCATTER_ELE_PER_BATCH threads will cooperate
    	//How to cooperate?
    	//Each threads read their own consecutive part, check how many same keys
    	
    	//initiate the localHis array
    	for(uint i = 0; i < RADIX; i++)	sharedInfo[tile_in_block].localHis[i * SCATTER_TILE_THREAD_NUM + lid_in_tile] = 0;
    	__syncthreads();

    	//doing the per-batch histogram counting
    	for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
    		//PAY ATTENTION: Here the shared memory access pattern has changed!!!!!!!
    		//instead for coalesced access, here each thread processes consecutive area of 
    		//SCATTER_ELE_PER_THREAD elements
    		const uint lo_id = lid_in_tile * SCATTER_ELE_PER_THREAD + i;
    		const unsigned char digit = sharedInfo[tile_in_block].digits[lo_id];
    		address_ele_per_thread[i] = digit * SCATTER_TILE_THREAD_NUM + lid_in_tile;
    		num_of_former_same_keys[i] = sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]];
    		sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]] = num_of_former_same_keys[i] + 1;
    	}
    	__syncthreads();

    	//now what have been saved?
    	//1. keys: the keys for this batch
    	//2. digits: the digits for this batch
    	//3. address_ele_per_thread: the address in localCount for each element visited by a thread
    	//4. num_of_former_same_keys: # of same keys before this key
    	//5. localHis: storing the key counts

    	//localHist structure:
    	//[SCATTER_TILE_THREAD_NUM for Radix 0][SCATTER_TILE_THREAD_NUM for Radix 1]...

    	//now exclusive scan the localHist:
//doing the naive scan:--------------------------------------------------------------------------------------------------------------------------------
    	int digitCount = 0;

    	if (lid_in_tile < RADIX) {
    		uint localBegin = lid_in_tile * SCATTER_TILE_THREAD_NUM;
    		unsigned char prev = sharedInfo[tile_in_block].localHis[localBegin];
    		unsigned char now = 0;
    		sharedInfo[tile_in_block].localHis[localBegin] = 0;
    		for(int i = localBegin + 1; i < localBegin + SCATTER_TILE_THREAD_NUM; i++) {
    			now = sharedInfo[tile_in_block].localHis[i];
    			sharedInfo[tile_in_block].localHis[i] = sharedInfo[tile_in_block].localHis[i-1] + prev;
    			prev = now;
    			if (i == localBegin + SCATTER_TILE_THREAD_NUM - 1)	sharedInfo[tile_in_block].countArr[lid_in_tile] = sharedInfo[tile_in_block].localHis[i] + prev;
    		}
    	}
    	__syncthreads();

    	if (lid_in_tile < RADIX)	digitCount = sharedInfo[tile_in_block].countArr[lid_in_tile];

    	if (lid_in_tile == 0) {
    		//exclusive scan for the countArr
    		unsigned char prev = sharedInfo[tile_in_block].countArr[0];
    		unsigned char now = 0;
    		sharedInfo[tile_in_block].countArr[0] = 0;
    		for(uint i = 1; i < RADIX; i++) {
    			now = sharedInfo[tile_in_block].countArr[i];
    			sharedInfo[tile_in_block].countArr[i] = sharedInfo[tile_in_block].countArr[i-1] + prev;
    			prev = now;
    		}
    	}
    	__syncthreads();

    	if ( lid_in_tile < RADIX) {
    		//scan add back
    		uint localBegin = lid_in_tile * SCATTER_TILE_THREAD_NUM;
    		for(uint i = localBegin; i < localBegin + SCATTER_TILE_THREAD_NUM; i++)
    			sharedInfo[tile_in_block].localHis[i] += sharedInfo[tile_in_block].countArr[lid_in_tile];

    		//now consider the offsets:
    		//lid_in_tile which is < RADIX stores the global offset for this digit in this tile
    		//here: updating the global offset
    		//PAY ATTENTION: Why offset needs to deduct countArr? See the explaination in the final scatter!!
    		sharedInfo[tile_in_block].bias[lid_in_tile] = offset - sharedInfo[tile_in_block].countArr[lid_in_tile];
    		offset += digitCount;

    	}

//end of naive scan:-------------------------------------------------------------------------------------------------------------------------------------

    	// if (lid_in_tile < RADIX) {
    	// 	sharedInfo[tile_in_block].bias[lid_in_tile] = offset;
    	// 	offset += digitCount;
    	// }

    	//still consecutive access!!
    	for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
    		const unsigned char lo_id = lid_in_tile * SCATTER_ELE_PER_THREAD + i;

    		//position of this element(with id: lo_id) being scattered to
    		uint pos = num_of_former_same_keys[i] + sharedInfo[tile_in_block].localHis[address_ele_per_thread[i]];

    		//since this access pattern is different from the scatter pattern(coalesced access), the position should be stored
    		//also because this lo_id is not tractable in the scatter, thus using pos as the index instead of lo_id!!
    		// both pos and lo_id are in the range of [0, SCATTER_ELE_PER_BATCH)
    		sharedInfo[tile_in_block].shuffle[pos] = lo_id;		
    	}
    	__syncthreads();

    	//scatter back to the global memory, iterating in the shuffle array
    	for(uint i = 0; i < SCATTER_ELE_PER_THREAD; i++) {
    		const uint lo_id = lid_in_tile + i * SCATTER_TILE_THREAD_NUM;	//coalesced access
    		if ((int)lo_id < (int)end - (int)start) {						//in case that some threads have been larger than the total length causing index overflow
    			const unsigned char position = sharedInfo[tile_in_block].shuffle[lo_id];	//position is the lo_id above
    			const unsigned char myDigit = sharedInfo[tile_in_block].digits[position];	//when storing digits, the storing pattern is lid_in_tile + i * SCATTER_TILE_THREAD_NUM, 

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
                // if (addr < 0)   printf("block %d,id %d: addr %d\n",blockId, localId,addr);
                // else            printf("correct: block %d,id %d: addr %d\n",blockId, localId,addr);
    			d_out[addr] = sharedInfo[tile_in_block].keys[position];
    		}
    	}
    	__syncthreads();
    }
}


void testRadixSort() { 
	// int len = 16777216;
	int len = 16000000;

	int *h_source = new int[len];

	for(int i = 0; i < len; i++)	h_source[i] = rand() % INT_MAX;

	int blockLen = REDUCE_BLOCK_SIZE * REDUCE_ELE_PER_THREAD;
	int gridSize = (len + blockLen- 1) / blockLen;

	int *h_his = new int[gridSize * RADIX];
	//histogram
	int *histogram, *d_source, *d_temp;
	checkCudaErrors(cudaMalloc(&d_source, sizeof(int)* len));
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(int)* len));
	checkCudaErrors(cudaMalloc(&histogram, sizeof(int)* gridSize * RADIX));

	int *h_dest = new int[len];

	dim3 grid(gridSize);
	dim3 block(REDUCE_BLOCK_SIZE);

	int expeTime = 10;
	float totalTime = 0.0f;
	bool res = true;
	for(int i = 0; i < expeTime; i++) {

		cudaMemcpy(d_source, h_source, sizeof(int) * len, cudaMemcpyHostToDevice);

		float myTime = 0.0f;

		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		for(int shiftBits = 0; shiftBits < sizeof(int) * 8; shiftBits += BITS) {
			radix_reduce<<<grid, block, sizeof(int) * REDUCE_BLOCK_SIZE * RADIX>>>(d_source, len, blockLen, histogram, shiftBits);
			scan(histogram, gridSize * RADIX, 1, 1024);
			int tileLen = REDUCE_BLOCK_SIZE * REDUCE_ELE_PER_THREAD;
			radix_scatter<<<(gridSize+SCATTER_TILES_PER_BLOCK-1)/SCATTER_TILES_PER_BLOCK,SCATTER_BLOCK_SIZE>>>( d_source, d_temp, len, tileLen, gridSize, histogram, shiftBits);
			int *temp = d_temp;
			d_temp = d_source;
			d_source = temp;
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);
	    cudaEventElapsedTime(&myTime,start, end);

		cudaMemcpy(h_dest, d_source, sizeof(int) * len, cudaMemcpyDeviceToHost);
		if (i != 0)		totalTime += myTime;
		//checking
		for(int i = 0; i < len-1; i++) {
	        if(h_dest[i] > h_dest[i+1]) {
				res = false;
	        }
		}
		if (!res)	break;
	}

    if (res)	std::cout<<"Correct!"<<std::endl;
    else 		std::cout<<"Wrong!"<<std::endl;
    cout<<"Avg time: "<<totalTime/(expeTime-1)<<" ms."<<endl;
	// thrust::device_ptr<int> dev_his(his);
	// thrust::device_ptr<int> dev_res_his(res_his);

	// gettimeofday(&start,NULL);

	// for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {

	// 	countHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
	// 	scanDevice(his, globalSize*RADIX, 1024, 1024,1);
	// 	writeHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
	// 	scatterDevice_int(d_source,d_temp, r_len, loc, 1024,32768);
	// 	int *swapPointer = d_temp;
	// 	d_temp = d_source;
	// 	d_source = swapPointer;
	// }
	// cudaDeviceSynchronize();
	// gettimeofday(&end,NULL);
	// totalTime = diffTime(end, start);

	// checkCudaErrors(cudaFree(his));
	// checkCudaErrors(cudaFree(d_temp));
	// checkCudaErrors(cudaFree(loc));


	delete[] h_source;
	delete[] h_dest;
}


/*******************************   end of fast radix sort ***************************************/

double radixSortDevice(Record *d_source, int r_len, int blockSize, int gridSize) {
	blockSize = 512;
	gridSize = 256;

	double totalTime = 0.0f;
	int globalSize = blockSize * gridSize;

	//histogram
	int *his, *loc, *res_his;
	checkCudaErrors(cudaMalloc(&his, sizeof(int)*globalSize * RADIX));
	checkCudaErrors(cudaMalloc(&loc, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&res_his, sizeof(int)*globalSize * RADIX));

	Record *d_temp;
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(Record)*r_len));

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	thrust::device_ptr<int> dev_his(his);
	thrust::device_ptr<int> dev_res_his(res_his);

	gettimeofday(&start,NULL);
	for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {

		countHis<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
		scanDevice(his, globalSize*RADIX, 1024, 1024,1);
		writeHis<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
		// scatter(d_source,d_temp, r_len, loc, 1024,32768);
		cudaMemcpy(d_source, d_temp, sizeof(Record)*r_len, cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);
	totalTime = diffTime(end, start);

	checkCudaErrors(cudaFree(his));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(loc));

	return totalTime;
}

double radixSortDevice_int(int *d_source, int r_len, int blockSize, int gridSize) {
	blockSize = 512;
	gridSize = 2048;

	double totalTime = 0.0f;
	int globalSize = blockSize * gridSize;

	//histogram
	int *his, *loc, *res_his;
	checkCudaErrors(cudaMalloc(&his, sizeof(int)*globalSize * RADIX));
	checkCudaErrors(cudaMalloc(&loc, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&res_his, sizeof(int)*globalSize * RADIX));

	int *d_temp;
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(int)*r_len));

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	thrust::device_ptr<int> dev_his(his);
	thrust::device_ptr<int> dev_res_his(res_his);

	gettimeofday(&start,NULL);

	std::cout<<"shared momery size:"<<sizeof(int)*RADIX*blockSize<<std::endl;
	for(int shiftBits = 0; shiftBits < sizeof(int)*8; shiftBits += BITS) {

		countHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source, r_len, his, shiftBits);
		scanDevice(his, globalSize*RADIX, 1024, 1024,1);
		writeHis_int<<<grid,block,sizeof(int)*RADIX*blockSize>>>(d_source,r_len,his,loc,shiftBits);
		// scatter(d_source,d_temp, r_len, loc, 1024,32768);
		int *swapPointer = d_temp;
		d_temp = d_source;
		d_source = swapPointer;
	}
	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);
	totalTime = diffTime(end, start);

	checkCudaErrors(cudaFree(his));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(loc));

	return totalTime;
}

double radixSortImpl(Record *h_source, int r_len, int blockSize, int gridSize) {
	double totalTime = 0.0f;
	Record *d_source;
	
	//thrust test
	int *keys = new int[r_len];
	int *values = new int[r_len];

	for(int i = 0; i < r_len; i++) {
		keys[i] = h_source[i].x;
		values[i] = h_source[i].y;
	}

	checkCudaErrors(cudaMalloc(&d_source, sizeof(Record)*r_len));
	cudaMemcpy(d_source, h_source, sizeof(Record)*r_len, cudaMemcpyHostToDevice);

	totalTime = radixSortDevice(d_source, r_len, blockSize, gridSize);

	cudaMemcpy(h_source, d_source, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(d_source));


	// struct timeval start, end;

	// gettimeofday(&start, NULL);
	// thrust::sorting::stable_radix_sort_by_key(values, values+r_len, keys);
	// gettimeofday(&end, NULL);

	// for(int i = 0; i < r_len; i++) {
	// 	cout<<h_source[i].x<<' '<<h_source[i].y<<'\t'<<keys[i]<<' '<<values[i]<<endl;
	// }
	// double thrustTime = diff(end,start);
	// cout<<"Thrust time for radixsort: "<<thrustTime<<" ms."<<endl;

	delete[] keys;
	delete[] values;
	return totalTime;
}

double radixSortImpl_int(int *h_source, int r_len, int blockSize, int gridSize) {
	double totalTime = 0.0f;
	int *h_thrust_source = new int[r_len];
	
	int *d_source;
	int *d_thrust_source;

	checkCudaErrors(cudaMalloc(&d_source, sizeof(int)*r_len));
	checkCudaErrors(cudaMalloc(&d_thrust_source, sizeof(int)*r_len));

	cudaMemcpy(d_source, h_source, sizeof(int)*r_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_thrust_source, h_source, sizeof(int)*r_len, cudaMemcpyHostToDevice);

	totalTime = radixSortDevice_int(d_source, r_len, blockSize, gridSize);

	cudaMemcpy(h_source, d_source, sizeof(int)*r_len, cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(d_source));

	struct timeval start, end;

	thrust::device_ptr<int> dev_source(d_thrust_source);

	gettimeofday(&start, NULL);
	thrust::sort(dev_source, dev_source+r_len);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	cudaMemcpy(h_thrust_source, d_thrust_source, sizeof(int) * r_len, cudaMemcpyDeviceToHost);
	
	double thrustTime = diffTime(end,start);
	std::cout<<"Thrust time for radixsort: "<<thrustTime<<" ms."<<std::endl;
	
	//check the thrust output with implemented output
	for(int i = 0; i < r_len; i++) {
		if (h_source[i] != h_thrust_source[i])	std::cerr<<"different"<<std::endl;
	}
	delete[] h_thrust_source;
	return totalTime;
}