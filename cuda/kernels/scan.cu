//
//  scan.cu
//  gpuqp_cuda
//
//  Created by Bryan on 01/21/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "kernels.h"
#include "dataDef.h"
using namespace std;

/*****************************************************************************/
/*
 *
 *              warp-wise scan
 *
 */


//scan_warp: d_source is always been inclusively scanned, the return number is the right output
//return: corresponding data of each thread index
//here d_source is already shared memory source
//length <= BLOCKSIZE
template<typename T> 
 __device__ T scan_warp_warpwise(T *d_source, const uint length, int isExclusive)
{
    int localId = threadIdx.x;
    const unsigned int lane = localId & SCAN_MASK;        //warp size is 32

    if (localId >= length)  return 0;   

    if (lane >= 1)      d_source[localId] += d_source[localId-1];
    if (lane >= 2)      d_source[localId] += d_source[localId-2];
    if (lane >= 4)      d_source[localId] += d_source[localId-4];
    if (lane >= 8)      d_source[localId] += d_source[localId-8];
    if (lane >= 16)     d_source[localId] += d_source[localId-16];

    if (isExclusive == 0)   return d_source[localId];
    else                    return (lane > 0)? d_source[localId-1] : (T)0;
}

//each block can process up to 1024*8=8192 elements if 1024 threads are used
//return: the sum of this block of data(only true when it is a inclusive scan)
template<typename T>
__global__ void scan_block_warpwise(T *d_source, const uint length, int isExclusive, bool isWriteSum, T *blockSum)
{
    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

    const unsigned int lane = localId & SCAN_MASK;
    const unsigned int warpId = localId >> SCAN_BITS;

    // extern __shared__ int share[];
    __shared__ T temp[SCAN_MAX_BLOCKSIZE*SCAN_ELEMENT_PER_THREAD];                  //global data used in this block
    __shared__ T warpSum[SCAN_MAX_BLOCKSIZE];      
    __shared__ T sumWarpSum[SCAN_MAX_WARP_NUM];

    //global mem to shared mem, coalesced access
    int startIdx = blockId * blockSize * SCAN_ELEMENT_PER_THREAD;
    for(int i = localId; i < SCAN_ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        if (startIdx + i >= length) break;
        temp[i] = d_source[startIdx + i];
    }
    __syncthreads();

    //endPlace: ending index of the part this block process
    int endPlace = (blockId+1)*blockSize*SCAN_ELEMENT_PER_THREAD >= length?  length : (blockId+1)*blockSize*SCAN_ELEMENT_PER_THREAD;
    T endEle = 0;
    if ( isWriteSum && (localId == 0) && isExclusive)   endEle = d_source[endPlace-1];
    __syncthreads();

    //doing the local scan: each thread does a sequential scan
    for(int i = localId * SCAN_ELEMENT_PER_THREAD + 1; i < (localId + 1) * SCAN_ELEMENT_PER_THREAD ; i++) {
        temp[i] += temp[i - 1];
    }
    warpSum[localId] = temp[(localId+1) * SCAN_ELEMENT_PER_THREAD - 1];
    __syncthreads();

    int warpVal =  scan_warp_warpwise(warpSum, blockSize, 1);                             //exclusive
    __syncthreads();

    if(lane == SCAN_MASK)    sumWarpSum[warpId] = warpSum[localId];
    __syncthreads();

    scan_warp_warpwise(sumWarpSum, blockSize/SCAN_WARPSIZE, 0);           //inclusive: 1024/32 = 32warps
    __syncthreads();
    
    if (warpId > 0)     warpVal += sumWarpSum[warpId-1]; 
    __syncthreads();

    //write back to the global mem
    for(int i = 0; i < SCAN_ELEMENT_PER_THREAD ; i++) {
        int currentLocalId = localId * SCAN_ELEMENT_PER_THREAD + i;
        int currentGlobalId = globalId * SCAN_ELEMENT_PER_THREAD + i;
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
/*************************** End of warpwise scan ***************************************/

/*************************** Blelloch scan ***************************************/
// #undef SCAN_ELEMENT_PER_THREAD
// #define SCAN_ELEMENT_PER_THREAD (2)
//local blelloch scan, by default length = 2^n
//here d_source is already shared memory source
//length <= BLOCKSIZE
template<typename T> 
 __device__ void scan_ble_small(T *d_source, const uint length, int isExclusive)
{
    int localId = threadIdx.x;
    if (localId >= length)  return;
    
    int offset = 1;                         //offset: the distance of the two added numbers
    //reduce
    for(int d = length >> 1; d > 0; d >>=1) {
        if (localId < d) {
            int ai = offset * ( 2 * localId + 1 ) - 1;
            int bi = offset * ( 2 * localId + 2 ) - 1;
            d_source[bi] += d_source[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    
    if (localId == 0)    {
        if (isExclusive == 1)
            d_source[length-1] = 0;
        else
            offset >>= 1;
    }
    __syncthreads();
    
    //sweep down
    if (isExclusive == 1) {
        for(int d = 1; d < length; d <<= 1) {
            offset >>= 1;
            if (localId < d) {
                int ai = offset * (2 * localId + 1) -1;
                int bi = offset * (2 * localId + 2) -1;
                
                int t = d_source[ai];
                d_source[ai] = d_source[bi];
                d_source[bi] += t;
            }
            __syncthreads();
        }
    }
    else {
        for(int d = 2; d <= length/2; d<<=1) {
            offset >>= 1;
            if (localId > 0 && localId < d) {
                int ai = localId * length / d - 1;
                int bi = offset + ai;
                d_source[bi] += d_source[ai];
            }
            __syncthreads();
        }
    }
}

//each block can process up to 1024*8=8192 elements if 1024 threads are used
//return: the sum of this block of data(only true when it is a inclusive scan)
template<typename T>
__global__ void scan_ble_large(T *d_source, const uint length, int isExclusive, bool isWriteSum, T *blockSum)
{
    int localId = threadIdx.x;
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

    // extern __shared__ int share[];
    __shared__ T temp[SCAN_MAX_BLOCKSIZE*SCAN_ELEMENT_PER_THREAD];                  //global data used in this block
    __shared__ T warpSum[SCAN_MAX_BLOCKSIZE];      

    //global mem to shared mem, coalesced access
    int startIdx = blockId * blockSize * SCAN_ELEMENT_PER_THREAD;
    for(int i = localId; i < SCAN_ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        if (startIdx + i >= length) break;
        temp[i] = d_source[startIdx + i];
    }
    __syncthreads();

    //endPlace: ending index of the part this block process
    int endPlace = (blockId+1)*blockSize*SCAN_ELEMENT_PER_THREAD >= length?  length : (blockId+1)*blockSize*SCAN_ELEMENT_PER_THREAD;
    T endEle = 0;
    if ( isWriteSum && (localId == 0) && isExclusive)   endEle = d_source[endPlace-1];
    __syncthreads();

    //doing the local scan: each thread does a sequential scan
    for(int i = localId * SCAN_ELEMENT_PER_THREAD + 1; i < (localId + 1) * SCAN_ELEMENT_PER_THREAD ; i++) {
        temp[i] += temp[i - 1];
    }
    warpSum[localId] = temp[(localId+1) * SCAN_ELEMENT_PER_THREAD - 1];
    __syncthreads();

    scan_ble_small(warpSum, blockSize, 1);      //exclusive scan

    //write back to the global mem
    for(int i = 0; i < SCAN_ELEMENT_PER_THREAD ; i++) {
        int currentLocalId = localId * SCAN_ELEMENT_PER_THREAD + i;
        int currentGlobalId = globalId * SCAN_ELEMENT_PER_THREAD + i;
        if (currentGlobalId >= length)  break;
        if (isExclusive == 0) 
            d_source[currentGlobalId] = temp[currentLocalId] + warpSum[localId];

        else
            d_source[currentGlobalId] = temp[currentLocalId] + warpSum[localId] - d_source[currentGlobalId];
    }   
    __syncthreads();

    //write the block sum
    if ( isWriteSum && (localId == 0))  {
        blockSum[blockId] = d_source[endPlace - 1] + endEle;
    }
}
/*************************** End of Blelloch scan ***************************************/

template<typename T>
__global__ void scan_addBlock(T* d_source, uint length, T* blockSum)
{
    int localId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    
    int startIdx = blockId * blockSize * SCAN_ELEMENT_PER_THREAD;
    int thisBlockSum = 0;
    if (blockId > 0)    thisBlockSum = blockSum[blockId-1];

    for(int i = localId; i < SCAN_ELEMENT_PER_THREAD * blockSize ; i += blockSize) {
        d_source[startIdx + i] += thisBlockSum;
    }
}

//can process up to SCAN_ELEMENT_PER_THREAD * BLOCKSIZE * GRIDSIZE * GRIDSIZE = 1024 * 8 * 1024 * 1024 data
template<typename T>
float scan_warpwise(T *d_source, int length, int isExclusive, int blockSize)
{

    float totalTime = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int element_per_block = blockSize * SCAN_ELEMENT_PER_THREAD;
    //decide how many levels should we handle(at most 3 levels: 8192^3)
    int firstLevelBlockNum = (length + element_per_block - 1 )/ element_per_block;
    int secondLevelBlockNum = (firstLevelBlockNum + element_per_block - 1) / element_per_block;
    int thirdLevelBlockNum = (secondLevelBlockNum + element_per_block - 1) / element_per_block;

    //length should be less than element_per_block^3
    assert(thirdLevelBlockNum == 1);    

    dim3 block(blockSize);

    cudaEventRecord(start);
    if (firstLevelBlockNum == 1) {      //length <= element_per_block, only 1 level is enough
        scan_block_warpwise<T><<<1, block>>>(d_source, length, isExclusive, false, NULL);
    }
    else if (secondLevelBlockNum == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
        dim3 grid1(firstLevelBlockNum);
        T *firstBlockSum;
        checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(T)*firstLevelBlockNum));
        scan_block_warpwise<T><<<grid1, block>>>(d_source, length, isExclusive, true, firstBlockSum);
        scan_block_warpwise<T><<<1, block>>>(firstBlockSum, firstLevelBlockNum, 0, false, NULL);
        scan_addBlock<T><<<grid1, block>>>(d_source, length, firstBlockSum);
        checkCudaErrors(cudaFree(firstBlockSum));
    }
    else {                              //length <= element_per_block^3, 3 levels are enough
        dim3 grid1(firstLevelBlockNum);
        dim3 grid2(secondLevelBlockNum);

        T *firstBlockSum, *secondBlockSum;
        checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(T)*firstLevelBlockNum));
        checkCudaErrors(cudaMalloc(&secondBlockSum, sizeof(T)*secondLevelBlockNum));

        scan_block_warpwise<T><<<grid1, block>>>(d_source, length, isExclusive, true, firstBlockSum);
        scan_block_warpwise<T><<<grid2, block>>>(firstBlockSum, firstLevelBlockNum, 0, true, secondBlockSum);
        scan_block_warpwise<T><<<1, block>>>(secondBlockSum, secondLevelBlockNum, 0, false, NULL);
        scan_addBlock<T><<<grid2, block>>>(firstBlockSum, firstLevelBlockNum, secondBlockSum);
        scan_addBlock<T><<<grid1, block>>>(d_source, length, firstBlockSum);

        checkCudaErrors(cudaFree(firstBlockSum));
        checkCudaErrors(cudaFree(secondBlockSum));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&totalTime, start, end);

    return totalTime;
}

//templates
template float scan_warpwise<int>(int *d_source, int length, int isExclusive, int blockSize);
template float scan_warpwise<long>(long *d_source, int length, int isExclusive, int blockSize);
template float scan_warpwise<float>(float *d_source, int length, int isExclusive, int blockSize);
template float scan_warpwise<double>(double *d_source, int length, int isExclusive, int blockSize);


//can process up to SCAN_ELEMENT_PER_THREAD * BLOCKSIZE * GRIDSIZE * GRIDSIZE = 1024 * 8 * 1024 * 1024 data
template<typename T>
float scan_ble(T *d_source, int length, int isExclusive, int blockSize)
{
// T *another_d_source;
// cudaMemcpy(another_d_source, d_source, sizeof(T) * length, cudaMemcpyDeviceToDevice);
    float totalTime = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int element_per_block = blockSize * SCAN_ELEMENT_PER_THREAD;
    //decide how many levels should we handle(at most 3 levels: 8192^3)
    int firstLevelBlockNum = (length + element_per_block - 1 )/ element_per_block;
    int secondLevelBlockNum = (firstLevelBlockNum + element_per_block - 1) / element_per_block;
    int thirdLevelBlockNum = (secondLevelBlockNum + element_per_block - 1) / element_per_block;

    //length should be less than element_per_block^3
    assert(thirdLevelBlockNum == 1);    

    dim3 block(blockSize);

    cudaEventRecord(start);
    if (firstLevelBlockNum == 1) {      //length <= element_per_block, only 1 level is enough
        scan_ble_large<T><<<1, block>>>(d_source, length, isExclusive, false, NULL);
    }
    else if (secondLevelBlockNum == 1) {    //lenth <= element_per_block^2, 2 levels are needed.
        dim3 grid1(firstLevelBlockNum);
        T *firstBlockSum;
        checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(T)*firstLevelBlockNum));
        scan_ble_large<T><<<grid1, block>>>(d_source, length, isExclusive, true, firstBlockSum);
        scan_ble_large<T><<<1, block>>>(firstBlockSum, firstLevelBlockNum, 0, false, NULL);
        scan_addBlock<T><<<grid1, block>>>(d_source, length, firstBlockSum);
        checkCudaErrors(cudaFree(firstBlockSum));
    }
    else {                              //length <= element_per_block^3, 3 levels are enough
        dim3 grid1(firstLevelBlockNum);
        dim3 grid2(secondLevelBlockNum);

        T *firstBlockSum, *secondBlockSum;
        checkCudaErrors(cudaMalloc(&firstBlockSum, sizeof(T)*firstLevelBlockNum));
        checkCudaErrors(cudaMalloc(&secondBlockSum, sizeof(T)*secondLevelBlockNum));

        scan_ble_large<T><<<grid1, block>>>(d_source, length, isExclusive, true, firstBlockSum);
        scan_ble_large<T><<<grid2, block>>>(firstBlockSum, firstLevelBlockNum, 0, true, secondBlockSum);
        scan_ble_large<T><<<1, block>>>(secondBlockSum, secondLevelBlockNum, 0, false, NULL);
        scan_addBlock<T><<<grid2, block>>>(firstBlockSum, firstLevelBlockNum, secondBlockSum);
        scan_addBlock<T><<<grid1, block>>>(d_source, length, firstBlockSum);

        checkCudaErrors(cudaFree(firstBlockSum));
        checkCudaErrors(cudaFree(secondBlockSum));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&totalTime, start, end);

// scanImpl(another_d_source, length, blockSize, GRIDSIZE, isExclusive);

    return totalTime;
}

//templates
template float scan_ble<int>(int *d_source, int length, int isExclusive, int blockSize);
template float scan_ble<long>(long *d_source, int length, int isExclusive, int blockSize);
template float scan_ble<float>(float *d_source, int length, int isExclusive, int blockSize);
template float scan_ble<double>(double *d_source, int length, int isExclusive, int blockSize);

/*****************************************************************************/
/********************************* host functions ****************************/
//calculate the smallest power of two larger than input

#undef SCAN_ELEMENT_PER_THREAD
#define SCAN_ELEMENT_PER_THREAD (2)

__device__
int ceilPowerOfTwo(uint input) {
    int k = 1;
    while (k < input) k<<=1;
    return k;
}

//each block can proceed BLOCKSIZE*2 (currently 1024*2) numbers
__global__
void scan_small(int* records,       //size: number of elements
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
void scan_large(int* records,           //size: number of elements
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

// __global__
// void addBlock(int* records,
//               uint length,
//               int* blockSum)
// {
//     extern __shared__ int temp[];
//     int localId = threadIdx.x;
//     int globalId = blockIdx.x * blockDim.x + threadIdx.x;
//     int groupId = blockIdx.x;
    
//     temp[2*localId] = 0;
//     temp[2*localId+1] = 0;
    
//     __syncthreads();
    
//     //memory copy
//     if (2*globalId<length)    temp[2*localId] = records[2*globalId];
//     if (2*globalId+1<length)  temp[2*localId+1] = records[2*globalId+1];
    
//     __syncthreads();
    
//     int thisBlockSum = 0;
    
//     if (groupId > 0) {
//         thisBlockSum = blockSum[groupId-1];
//         if (2*globalId < length)        temp[2*localId] += thisBlockSum;
//         if (2*globalId + 1 < length)    temp[2*localId + 1] += thisBlockSum;
//     }
//     __syncthreads();
    
//     if (2*globalId < length)        records[2*globalId] = temp[2*localId];
//     if (2*globalId + 1 < length)    records[2*globalId + 1] = temp[2*localId+1];
// }

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
    scan_large<<<firstGrid,block,tempBitSize>>>(d_source, r_len, isExclusive, firstBlockSum);
    scan_large<<<secondGrid,block,tempBitSize >>>(firstBlockSum, firstLevelBlockNum, 0, secondBlockSum);
    scan_small<<<thirdGrid,block,tempBitSize>>>(secondBlockSum, secondLevelBlockNum, 0);
    scan_addBlock<int><<<secondGrid, block>>>(firstBlockSum, firstLevelBlockNum, secondBlockSum);
    scan_addBlock<int><<<firstGrid, block>>>(d_source, r_len, firstBlockSum);
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