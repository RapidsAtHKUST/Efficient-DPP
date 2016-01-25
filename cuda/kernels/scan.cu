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

void scanImpl(int *h_source, int r_len, 
			  int blockSize, int gridSize, double &time, int isExclusive)
{
	int *d_source;
    int *d_source_thrust, *d_dest_thrust;
    int *h_source_thrust = new int[r_len];

    int tempSize = blockSize * 2;
    int tempBitSize = tempSize * sizeof(int);

    int firstLevelBlockNum = ceil((double)r_len/tempSize);
    int secondLevelBlockNum = ceil((double)firstLevelBlockNum/tempSize);

    //set the local and global size
    dim3 block(blockSize);
    dim3 firstGrid(firstLevelBlockNum);
    dim3 secondGrid(secondLevelBlockNum);
    dim3 thirdGrid(1);

    checkCudaErrors(cudaMalloc(&d_source,sizeof(int)*r_len));
    cudaMalloc(&d_source_thrust, sizeof(int)*r_len);
    cudaMalloc(&d_dest_thrust, sizeof(int)*r_len);

    cudaMemcpy(d_source, h_source, sizeof(int) * r_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_thrust, h_source, sizeof(int)*r_len, cudaMemcpyHostToDevice);

    time = 0;
    
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

    time = diffTime(end,start);

    cudaMemcpy(h_source, d_source, sizeof(int) * r_len, cudaMemcpyDeviceToHost);

    //thrust::scan test
    thrust::device_ptr<int> ptr_d_source_thrust(d_source_thrust);
    thrust::device_ptr<int> ptr_d_dest_thrust(d_dest_thrust);

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
        if (h_source[i] != h_source_thrust[i])  thrustRes = false;
    }

    if (thrustRes)      cout<<"Same as thrust!"<<endl;
    else                cout<<"Different with thrust!"<<endl;
    cout<<"Thrust time: "<<thrustTime<<" ms."<<endl;
}