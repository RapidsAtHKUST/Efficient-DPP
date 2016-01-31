#include "kernels.h"

int getNearestLarger2Power(int input) {
    int k = 1;
    while (k < input)   k<<=1;
    return k;
}

__global__
void bitonicSort (Record* reSource,
					const int group,
                    const int length,
                    const int dir,
                    const int flip)
{
    int groupId = blockIdx.x;
    int groupNum = gridDim.x;
    int localId = threadIdx.x;
    int localSize = blockDim.x;

    for(int gpos = groupId; gpos < group; gpos += groupNum) {
        for(int pos = localId; pos < length/2; pos += localSize) {
            int begin = gpos * length;
            int delta;
            if (flip == 1)      delta = length - 1;
            else                delta = length/2;

            int a = begin + pos;
            int b = begin + delta - flip * pos;

            if ( dir == (reSource[a].y > reSource[b].y)) {
                Record temp = reSource[a];
                reSource[a] = reSource[b];
                reSource[b] = temp;
            }
        }
    }

}

__global__
void bitonicSort_op (Record* d_source,
                    const int r_len,
                    const int base,
                    const int interval)
{
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    int globalSize = blockDim.x * gridDim.x;

    for(int pos = globalId; pos < r_len; pos += globalSize) {
    	int compareIndex = pos^interval;
    	if (compareIndex > pos) {
    		Record thisRecord = d_source[pos];
    		Record compareRecord = d_source[compareIndex];

    		if ( (pos & base) == 0 && thisRecord.y > compareRecord.y) {
    			d_source[pos] = compareRecord;
    			d_source[compareIndex] = thisRecord;
    		}
    		else if ((pos & base) != 0 && thisRecord.y < compareRecord.y) {
				d_source[pos] = compareRecord;
    			d_source[compareIndex] = thisRecord;
    		}
    	}
    }
}

double bitonicSortDevice(Record *d_source, int r_len, int dir, int blockSize, int gridSize) {

	double totalTime = 0.0f;

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	//bitonic sort
    for(int i = 2; i <= r_len ; i <<= 1) {
        for(int j = i ; j > 1; j>>= 1) {
            int group = r_len/j;
            int flip = (j==i?1:-1);

            gettimeofday(&start, NULL);
            bitonicSort<<<grid,block>>>(d_source, group, j, dir, flip);
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            totalTime += diffTime(end, start);
        }
    }
    return totalTime;

}

double bitonicSortDevice_op(Record *d_source, int r_len, int dir, int blockSize, int gridSize) {

	double totalTime = 0.0f;

	dim3 grid(gridSize);
	dim3 block(blockSize);

	struct timeval start, end;

	//bitonic sort
    gettimeofday(&start, NULL);
    for(int base = 2; base <= r_len ; base <<= 1) {
        for(int interval = base>>1 ; interval > 0; interval>>= 1) {
            bitonicSort_op<<<grid,block>>>(d_source, r_len, base, interval);
        }
    }
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);

	totalTime += diffTime(end, start);
    return totalTime;
}

//dir: 1 for asc, 0 for des
double bitonicSortImpl(Record *h_source, int r_len, int dir, int blockSize, int gridSize) {
	
	double totalTime = 0.0f;

	//fill up the record array to be size of 2^n
    int ceilLength = getNearestLarger2Power(r_len);      //get the padded length

    int paddingInt;
    if (dir == 1)	paddingInt = INT_MAX;
    else			paddingInt = INT_MIN;

    Record *maxArr = NULL;
    if (ceilLength - r_len != 0) {
        maxArr = new Record[ceilLength];
        for(int i = 0; i < r_len; i++) {
        	maxArr[i] = h_source[i];
        }
        for(int i = r_len; i < ceilLength; i++) {
            maxArr[i].x = paddingInt;
            maxArr[i].y = paddingInt;
        }
    }
    else {
    	maxArr = h_source;
    }
    
    //memory allocation
    Record *d_source;
    checkCudaErrors(cudaMalloc(&d_source, sizeof(Record)*ceilLength));
    cudaMemcpy(d_source, maxArr, sizeof(Record)*ceilLength, cudaMemcpyHostToDevice);

    totalTime = bitonicSortDevice_op(d_source,ceilLength, dir, blockSize, gridSize);
    // totalTime = bitonicSortDevice(d_source,ceilLength, dir, blockSize, gridSize);

    //memory written back
    cudaMemcpy(h_source, d_source, sizeof(Record)*r_len, cudaMemcpyDeviceToHost);

    if (r_len != ceilLength)    delete[] maxArr;
    maxArr = NULL;

    return totalTime;
}