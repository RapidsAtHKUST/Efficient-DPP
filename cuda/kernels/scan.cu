//
//  scan.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/21/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"
using namespace std;


//calculate the smallest power of two larger than input
__device__
int ceilPowerOfTwo(uint input) {
    int k = 1;
    while (k < input) k<<=1;
    return k;
}

//each block can proceed BLOCKSIZE*2 (currently 1024) numbers
__global__
void prefixScan(int* records,       //size: number of elements
                uint length,
                uint isExclusive)
{
	extern __shared__ int temp[];

	int localId = threadIdx.x;
    
    temp[2*localId] = 0;                            //initialize to zero for 0 padding
    temp[2*localId + 1] = 0;
    __syncthreads();
    
    int offset = 1;                                 //offset: the distance of the two added numbers
    int paddedLength = ceilPowerOfTwo(length);      //padding
    
    //memory copy
    if (2*localId<length)    temp[2*localId] = records[2*localId];
    if (2*localId+1<length)  temp[2*localId+1] = records[2*localId+1];
    
    __syncthreads();
    
    //reduce
    for(int d = paddedLength >> 1; d > 0; d >>=1) {
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    
    if (localId == 0)    {
        temp[paddedLength-1] = 0;
    }
    __syncthreads();
    
    //sweep down
    for(int d = 1; d < paddedLength; d <<= 1) {
        offset >>= 1;
        if (localId < d) {
            int ai = offset * (2 * localId + 1) -1;
            int bi = offset * (2 * localId + 2) -1;
            
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        
        __syncthreads();
    }
    
    if (isExclusive == 1) {     //exclusive
        //memory output
        if (2*localId<length)    records[2*localId] = temp[2*localId];
        if (2*localId+1<length)  records[2*localId+1] = temp[2*localId+1];
    }
    else {                      //inclusive
        //memory output
        if (2*localId<length)    records[2*localId] += temp[2*localId];
        if (2*localId+1<length)  records[2*localId+1] += temp[2*localId+1];
    }
}

//scan large array: parittion into blocks, each block proceeds 1024 numbers.
__global__
void scanLargeArray(int* records,           //size: number of elements
                    uint length,
                    uint isExclusive,
                    int* blockSum)       //store the sums of each block
{
	extern __shared__ int temp[];

	int localId = threadIdx.x;
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int groupId = blockIdx.x;

    uint tempSize =  blockDim.x * 2;

    temp[2*localId] = 0;                            //initialize to zero for 0 padding
    temp[2*localId + 1] = 0;
    
    __syncthreads();
    
    int offset = 1;                                 //offset: the distance of the two added numbers
    
    //memory copy
    if (2*globalId<length)    temp[2*localId] = records[2*globalId];
    if (2*globalId+1<length)  temp[2*localId+1] = records[2*globalId+1];
    
    //reduce
    for(int d = tempSize >> 1; d > 0; d >>=1) {
        
        __syncthreads();
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    
    if (localId == 0)   {
        blockSum[groupId] = temp[tempSize-1];              //write the sum
        temp[tempSize-1] = 0;
    }
    
    __syncthreads();
    
    //sweep down
    for(int d = 1; d < tempSize; d <<= 1) {
        offset >>= 1;
        
        __syncthreads();
        if (localId < d) {
            int ai = offset * (2 * localId + 1) -1;
            int bi = offset * (2 * localId + 2) -1;
            
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    if (isExclusive == 1) {
        //memory output
        if (2*globalId<length)    records[2*globalId] = temp[2*localId];
        if (2*globalId+1<length)  records[2*globalId+1] = temp[2*localId+1];
    }
    else {
        //memory output
        if (2*globalId<length)    records[2*globalId] += temp[2*localId];
        if (2*globalId+1<length)  records[2*globalId+1] += temp[2*localId+1];
    }
}

__global__
void addBlock(int* records,
              uint length,
              int* blockSum)
{
	extern __shared__ int temp[];
	int localId = threadIdx.x;
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int groupId = blockIdx.x;
    
    temp[2*localId] = 0;
    temp[2*localId+1] = 0;
    
    __syncthreads();
    
    //memory copy
    if (2*globalId<length)    temp[2*localId] = records[2*globalId];
    if (2*globalId+1<length)  temp[2*localId+1] = records[2*globalId+1];
    
    __syncthreads();
    
    int thisBlockSum = 0;
    
    if (groupId > 0) {
        thisBlockSum = blockSum[groupId-1];
        if (2*globalId < length)        temp[2*localId] += thisBlockSum;
        if (2*globalId + 1 < length)    temp[2*localId + 1] += thisBlockSum;
    }
    __syncthreads();
    
    if (2*globalId < length)        records[2*globalId] = temp[2*localId];
    if (2*globalId + 1 < length)    records[2*globalId + 1] = temp[2*localId+1];
}

/*****************************************************************************/
/*
 *
 *              warp-wise scan
 *
 */

//warp size if 32(2^5)
#define BITS        (5)
#define WARPSIZE    (1<<BITS)
#define MASK        ((1<<BITS)-1)

#define ELEMENT_PER_THREAD (8)
#define MAX_BLOCKSIZE   (1024)      //a block can have at most 1024 threads

//scan_warp: d_source is always been inclusively scanned, the return number is the right output
//return: corresponding data of each thread index
__device__
int scan_warp(int *d_source, const uint length, int isExclusive)
{
    int localId = threadIdx.x;
    const unsigned int lane = localId & MASK;        //warp size is 32

    if (localId >= length)  return 0;   

    if (lane >= 1)      d_source[localId] += d_source[localId-1];
    if (lane >= 2)      d_source[localId] += d_source[localId-2];
    if (lane >= 4)      d_source[localId] += d_source[localId-4];
    if (lane >= 8)      d_source[localId] += d_source[localId-8];
    if (lane >= 16)     d_source[localId] += d_source[localId-16];

    if (isExclusive == 0)   return d_source[localId];
    else                    return (lane > 0)? d_source[localId-1] : 0;
}

//each block can process up to 1024*8=8192 elements if 1024 threads are used
//return: the sum of this block of data(only true when it is a inclusive scan)
__global__
void scan_block(int *d_source, const uint length, int isExclusive, bool isWriteSum, int *blockSum)
{
    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

    const unsigned int lane = localId & MASK;
    const unsigned int warpId = localId >> BITS;

    // extern __shared__ int share[];
    __shared__ int temp[MAX_BLOCKSIZE*ELEMENT_PER_THREAD];                  //global data used in this block
    __shared__ int warpSum[MAX_BLOCKSIZE];      
    __shared__ int sumWarpSum[WARPSIZE];

    //global mem to shared mem, coalesced access
    int startIdx = blockId * blockSize * ELEMENT_PER_THREAD;
    for(int i = localId; i < ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        if (startIdx + i >= length) break;
        temp[i] = d_source[startIdx + i];
    }
    __syncthreads();

    //endPlace: ending index of the part this block process
    int endPlace = (blockId+1)*blockSize*ELEMENT_PER_THREAD >= length?  length : (blockId+1)*blockSize*ELEMENT_PER_THREAD;
    int endEle = 0;
    if ( isWriteSum && (localId == 0) && isExclusive)   endEle = d_source[endPlace-1];
    __syncthreads();

    //doing the local scan
    for(int i = localId * ELEMENT_PER_THREAD + 1; i < (localId + 1) * ELEMENT_PER_THREAD ; i++) {
        temp[i] += temp[i - 1];
    }
    warpSum[localId] = temp[(localId+1) * ELEMENT_PER_THREAD - 1];
    __syncthreads();

    int warpVal = scan_warp(warpSum, blockSize, 1);                             //exclusive
    __syncthreads();

    if(lane == MASK)    sumWarpSum[warpId] = warpSum[localId];
    __syncthreads();

    if (warpId==0)      scan_warp(sumWarpSum, blockSize/WARPSIZE, 0);           //inclusive: 1024/32 = 32warps
    __syncthreads();
    
    if (warpId > 0)     warpVal += sumWarpSum[warpId-1]; 
    __syncthreads();

    //write back to the global mem
    for(int i = 0; i < ELEMENT_PER_THREAD ; i++) {
        int currentLocalId = localId * ELEMENT_PER_THREAD + i;
        int currentGlobalId = globalId * ELEMENT_PER_THREAD + i;
        if (currentGlobalId >= length)  break;
        if (isExclusive == 0) 
            d_source[currentGlobalId] = temp[currentLocalId] + warpVal;
        else
            d_source[currentGlobalId] = temp[currentLocalId] + warpVal - d_source[currentGlobalId];
    }   
    __syncthreads();

    //write the block sum
    if ( isWriteSum && (localId == 0))  {
        blockSum[blockId] = d_source[endPlace - 1] + endEle;
    }
}

__global__
void scan_addBlock(int* d_source, uint length, int* blockSum)
{
    __shared__ int temp[MAX_BLOCKSIZE * ELEMENT_PER_THREAD];
    int localId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    
    //memory copy
    int startIdx = blockId * blockSize * ELEMENT_PER_THREAD;
    for(int i = localId; i < ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        temp[i] = d_source[startIdx + i];
    }
    __syncthreads();
    
    //add block sum
    if (blockId > 0) {
        int thisBlockSum = blockSum[blockId-1];
        for(int i = localId * ELEMENT_PER_THREAD; i < (localId + 1) * ELEMENT_PER_THREAD ; i++) {
            temp[i] += thisBlockSum;
        }
        // if (localId == 0) printf("haha:%d\n", temp[0]);
    }
    __syncthreads();
    
    for(int i = localId; i < ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        d_source[startIdx + i] = temp[i];
    }
}

//can process up to ELEMENT_PER_THREAD * BLOCKSIZE * GRIDSIZE * GRIDSIZE = 1024 * 8 * 1024 * 1024 data
void scan_global(int *d_source, int length, int isExclusive, int blockSize)
{
    int element_per_block = blockSize * ELEMENT_PER_THREAD;
    //decide how many levels should we handle(at most 3 levels: 8192^3)
    int firstLevelBlockNum = ceil((double)length / element_per_block);
    int secondLevelBlockNum = ceil((double)firstLevelBlockNum / element_per_block);
    int thirdLevelBlockNum = ceil((double)secondLevelBlockNum / element_per_block);

    //length should be less than element_per_block^3
    assert(thirdLevelBlockNum == 1);    

    dim3 block(blockSize);

    if (firstLevelBlockNum == 1) {      //length <= element_per_block, only 1 level is enough
        scan_block<<<1, block>>>(d_source, length, isExclusive, false, NULL);
    }
    else if (secondLevelBlockNum == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
        dim3 grid1(firstLevelBlockNum);
        int *firstBlockSum;
        checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(int)*firstLevelBlockNum));
        scan_block<<<grid1, block>>>(d_source, length, isExclusive, true, firstBlockSum);
        scan_block<<<1, block>>>(firstBlockSum, firstLevelBlockNum, 0, false, NULL);
        scan_addBlock<<<grid1, block>>>(d_source, length, firstBlockSum);
    }
    else {                              //length <= element_per_block^3, 3 levels are enough
        //to to:
    }
}

//testing function for warp-wise scan
void scan_warp_test() {
    int totalLoop = 2;
    double totalTime = 0.0f;
    bool res = true;
    int isExclusive = 0;

    int len = 16000000;
    cout<<"Data size: "<<len<<endl;

    int *h_source = new int[len];
    int *h_dest = new int[len];
    int *h_source_com = new int[len];
    int *h_source_com_exclusive = new int[len];

    for(int i = 0; i < len; i++) {
        h_source[i] = i+1;
        h_source_com[i] = i+1;
    }

    //cpu inclusive
    for(int i = 0; i < len; i++)
        if (i>0)    h_source_com[i] += h_source_com[i-1];

    h_source_com_exclusive[0] = 0;
    for(int i = 1; i < len; i++) {
        h_source_com_exclusive[i] = h_source_com_exclusive[i-1] + h_source[i-1];
    }
    
    int *d_source;
    checkCudaErrors(cudaMalloc(&d_source, sizeof(int) * len));

    //cuda timing event
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for(int loop = 0; loop < totalLoop; loop++) {
        cudaMemcpy(d_source, h_source, sizeof(int) * len, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        scan_global(d_source,len,isExclusive,1024);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        cout<<"finish "<<loop+1<<endl;
        float myTime = 0.0f;
        cudaEventElapsedTime(&myTime,start, end);

        //get rid of the first loop: time is not accurate in the first run
        if (loop > 0)   totalTime += myTime;

        cudaMemcpy(h_dest, d_source, sizeof(int) * len, cudaMemcpyDeviceToHost);

        int *h_check;
        if (isExclusive == 1)   h_check = h_source_com_exclusive;
        else                    h_check = h_source_com;

        for(int i = 0; i < len; i++) {
            if (h_check[i] != h_dest[i])    
                res = false;                
        }
        if (!res)   break;

        isExclusive = 1 - isExclusive;
    }

    cout<<"Output: ";
    if (res)    cout<<"correct!"<<endl;
    else    {
        cout<<"incorrect ";
        if (isExclusive == 1)   cout<<"for exclusive."<<endl;
        else                    cout<<"for inclusive."<<endl;

        int *h_check;
        if (isExclusive==1) h_check = h_source_com_exclusive;
        else                h_check = h_source_com;

        for(int i = 0; i < len; i++) {
            if (h_check[i] != h_dest[i])    {
                cout<<i<<": "<<h_dest[i]<<'\t'<<h_check[i]<<"\tWrong!"<<endl;
            }
        }   
    }

    delete[] h_source;
    delete[] h_dest;
    delete[] h_source_com;
    checkCudaErrors(cudaFree(d_source));

    cout<<"Avg time: "<<totalTime/(totalLoop-1)<<" ms."<<endl;
}


/*****************************************************************************/
/********************************* host functions ****************************/


double scanDevice(int *d_source, int r_len, int blockSize, int gridSize, int isExclusive) {
    
    int tempSize = blockSize * 2;
    int tempBitSize = tempSize * sizeof(int);

    int firstLevelBlockNum = ceil((double)r_len/tempSize);
    int secondLevelBlockNum = ceil((double)firstLevelBlockNum/tempSize);

    //set the local and global size
    dim3 block(blockSize);
    dim3 firstGrid(firstLevelBlockNum);
    dim3 secondGrid(secondLevelBlockNum);
    dim3 thirdGrid(1);

    double totalTime = 0.0f;
    
    struct timeval start, end;
    
    //temp memory objects
    int *firstBlockSum, *secondBlockSum;

    checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(int)*firstLevelBlockNum));
    checkCudaErrors(cudaMalloc(&secondBlockSum, sizeof(int)*secondLevelBlockNum));

    gettimeofday(&start,NULL);
    scanLargeArray<<<firstGrid,block,tempBitSize>>>(d_source, r_len, isExclusive, firstBlockSum);
    scanLargeArray<<<secondGrid,block,tempBitSize >>>(firstBlockSum, firstLevelBlockNum, 0, secondBlockSum);
    prefixScan<<<thirdGrid,block,tempBitSize>>>(secondBlockSum, secondLevelBlockNum, 0);
    addBlock<<<secondGrid, block, tempBitSize>>>(firstBlockSum, firstLevelBlockNum, secondBlockSum);
    addBlock<<<firstGrid, block, tempBitSize>>>(d_source, r_len, firstBlockSum);
    cudaDeviceSynchronize();
    gettimeofday(&end,NULL);

    totalTime = diffTime(end,start);

    return totalTime;
}

double scanImpl(int *h_source, int r_len, 
			  int blockSize, int gridSize, int isExclusive)
{
    double totalTime = 0.0f;

    int *d_source;
    checkCudaErrors(cudaMalloc(&d_source,sizeof(int)*r_len));

    int *h_source_thrust = new int[r_len];          //host memory
    for(int i = 0; i < r_len; i++) h_source_thrust[i] = h_source[i];

    cudaMemcpy(d_source, h_source, sizeof(int) * r_len, cudaMemcpyHostToDevice);
    totalTime = scanDevice(d_source, r_len, blockSize, gridSize, isExclusive);
    cudaMemcpy(h_source, d_source, sizeof(int) * r_len, cudaMemcpyDeviceToHost);

    //thrust::scan test
    int *d_source_thrust, *d_dest_thrust;           //device memory
    
    cudaMalloc(&d_source_thrust, sizeof(int)*r_len);
    cudaMalloc(&d_dest_thrust, sizeof(int)*r_len);
    cudaMemcpy(d_source_thrust, h_source_thrust, sizeof(int)*r_len, cudaMemcpyHostToDevice);

    thrust::device_ptr<int> ptr_d_source_thrust(d_source_thrust);
    thrust::device_ptr<int> ptr_d_dest_thrust(d_dest_thrust);

    struct timeval start, end;
    
    if (isExclusive) {
        gettimeofday(&start, NULL);
        thrust::exclusive_scan(ptr_d_source_thrust, ptr_d_source_thrust + r_len, ptr_d_dest_thrust);
        gettimeofday(&end, NULL);
    }
    else {
        gettimeofday(&start, NULL);
        thrust::inclusive_scan(ptr_d_source_thrust, ptr_d_source_thrust + r_len, ptr_d_dest_thrust);
        gettimeofday(&end, NULL);
    }
    
    double thrustTime = diffTime(end,start);

    cudaMemcpy(h_source_thrust,d_dest_thrust, sizeof(int)*r_len, cudaMemcpyDeviceToHost);

    //checking with thrust
    bool thrustRes = true;
    for(int i = 0; i < r_len; i++) {
        if (h_source[i] != h_source_thrust[i])  {
            thrustRes = false;
            cout<<i<<' '<<h_source[i]<<' '<<h_source_thrust[i]<<endl;
        }
    }

    if (thrustRes)      cout<<"Same as thrust!"<<endl;
    else                cout<<"Different with thrust!"<<endl;
    cout<<"Thrust time: "<<thrustTime<<" ms."<<endl;

	return totalTime;
}