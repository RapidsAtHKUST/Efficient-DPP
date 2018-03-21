#include "test.h"
#define MEM_EXPR_TIME (10)

void testMem() {
    int blockSize = 1024, gridSize = 8192;
    int repeat = 80;                                     //for read test
	int in_length_read = blockSize * gridSize * repeat;  //shrink the localsize and gridsize
    int out_length_write = blockSize * gridSize;

    std::cout<<"Input data size(read test): "<<in_length_read<<" ("<<in_length_read* sizeof(int)/1024/1024<<"MB)"<<std::endl;
    std::cout<<"Output data size(write write): "<<out_length_write<<" ("<<out_length_write* sizeof(int)/1024/1024<<"MB)"<<std::endl;

    float readTime = 0.0, writeTime = 0.0;

//-------------------- read test ----------------------------
	//allocate for the host memory
    int *h_in = new int[in_length_read];
    for(int i = 0; i < in_length_read; i++) {
        h_in[i] = i;
    }

    int *d_in, *d_out;
	checkCudaErrors(cudaMalloc(&d_in,sizeof(int)*in_length_read));
	checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*blockSize*gridSize));
	cudaMemcpy(d_in, h_in, sizeof(int)*in_length_read, cudaMemcpyHostToDevice);
	
	//executing read
	for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = testMemRead(d_in, d_out, blockSize, gridSize);

        //throw away the first result
        if (i != 0)     readTime += tempTime;
    }
    //finish read test, free the input space
	checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

//-------------------------- write test ---------------------------------
	checkCudaErrors(cudaMalloc(&d_out,sizeof(int)*out_length_write));

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = testMemWrite(d_out, blockSize, gridSize);

        //throw away the first result
        if (i != 0)     writeTime += tempTime;
    }    

    readTime /= (MEM_EXPR_TIME - 1);
    writeTime /= (MEM_EXPR_TIME - 1);

    delete[] h_in;
    checkCudaErrors(cudaFree(d_out));

    double throughput_read = computeMem(blockSize*gridSize*repeat, sizeof(int), readTime);
    double throughput_write = computeMem(blockSize*gridSize, sizeof(int), writeTime);

    std::cout<<"Time for memory read(Repeat:"<<repeat<<"): "<<readTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput_read<<" GB/s"<<std::endl;

    std::cout<<"Time for memory write: "<<writeTime<<" ms."<<'\t'
        <<"Bandwidth: "<<throughput_write<<" GB/s"<<std::endl;
}