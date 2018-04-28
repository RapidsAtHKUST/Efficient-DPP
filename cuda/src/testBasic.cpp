#include "test.h"
#include "kernels.h"
#define MEM_EXPR_TIME (10)
using namespace std;

void testMem() {
    cudaError_t err;
    int blockSize = 1024, gridSize = 32768;
    int len = blockSize * gridSize;
    std::cout<<"Data size(Multiplication): "<<len<<" ("<<len* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    float mulTime = 0.0;

    int *h_in, *d_in, *d_out;
    h_in = new int[len];
    for(int i = 0; i < len; i++) {
        h_in[i] = i;
    }
    checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*len));
    checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*len));
    cudaMemcpy(d_in, h_in, sizeof(int)*len, cudaMemcpyHostToDevice);

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = mul(d_in, d_out, blockSize, gridSize);

        //throw away the first result
        if (i != 0)     mulTime += tempTime;
    }
    mulTime /= (MEM_EXPR_TIME - 1);

    delete[] h_in;
    checkCudaErrors(cudaFree(d_out));

    //both read and write
    double throughput = computeMem(blockSize*gridSize*2, sizeof(int), mulTime);

    std::cout<<"Time for multiplication: "<<mulTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput<<" GB/s"<<std::endl;

}