#include "test.h"
#define MEM_EXPR_TIME (20)

void testMem(const int localSize, const int gridSize, float& readTime, float& writeTime, float& mulTime, int repeat) {

	int input_length_read = localSize / 2 * gridSize / 2 * repeat;  //shrink the localsize and gridsize

    //for write and mul, no need to repeat
    int input_length_others = localSize * gridSize * 2; 
    int output_length = localSize * gridSize * 2;

    std::cout<<"Input data size(read): "<<input_length_read<<std::endl;
    std::cout<<"Input data size(write & mul): "<<input_length_others<<std::endl;
    std::cout<<"Output data size: "<<output_length<<std::endl;

    assert(input_length_read > 0);
    assert(input_length_others > 0);
    assert(output_length > 0);

	//allocate for the host memory
	int *h_source_values = new int[input_length_read];  
    int *h_dest_values = new int[output_length];
    
    for(int i = 0; i < input_length_read; i++) {
        h_source_values[i] = rand() % 10000;
    }

    readTime = 0.0; writeTime = 0.0; mulTime = 0.0;
	
	//-------------------------- read ---------------------------------
	int *d_source_values, *d_dest_values;
	checkCudaErrors(cudaMalloc(&d_source_values,sizeof(int)*input_length_read));
	checkCudaErrors(cudaMalloc(&d_dest_values,sizeof(int)*output_length));
	cudaMemcpy(d_source_values, h_source_values, sizeof(int) * input_length_read, cudaMemcpyHostToDevice);	
	
	//executing read, write, mul, triad
	for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = testMemRead(d_source_values, d_dest_values, localSize/2, gridSize/2);

        //throw away the first result
        if (i != 0)     readTime += tempTime;
    }   
    //finish read test, free the input space
	cudaMemcpy(h_dest_values, d_dest_values, sizeof(int)*output_length, cudaMemcpyDeviceToHost);	
	checkCudaErrors(cudaFree(d_source_values));
    
	//-------------------------- write ---------------------------------
	checkCudaErrors(cudaMalloc(&d_source_values,sizeof(int)*input_length_others));
	cudaMemcpy(d_source_values, h_source_values, sizeof(int) * input_length_others, cudaMemcpyHostToDevice);	

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = testMemWrite(d_dest_values, localSize, gridSize);

        //throw away the first result
        if (i != 0)     writeTime += tempTime;
    }    
    cudaMemcpy(h_dest_values, d_dest_values, sizeof(int)*output_length, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(d_source_values));
	checkCudaErrors(cudaFree(d_dest_values));

	//-------------------------- mul ---------------------------------
	int2 *d_source_values_2, *d_dest_values_2;
	checkCudaErrors(cudaMalloc(&d_source_values_2,sizeof(int)*input_length_others));
	checkCudaErrors(cudaMalloc(&d_dest_values_2,sizeof(int)*output_length));
	cudaMemcpy(d_source_values_2, h_source_values, sizeof(int) * input_length_others, cudaMemcpyHostToDevice);

    for(int i = 0; i < MEM_EXPR_TIME; i++) {
        float tempTime = testMemMul(d_source_values_2, d_dest_values_2, localSize, gridSize);

        //throw away the first result
        if (i != 0)     mulTime += tempTime;
    }    

    readTime /= ((MEM_EXPR_TIME - 1)*repeat);
    writeTime /= (MEM_EXPR_TIME - 1);
    mulTime /= (MEM_EXPR_TIME - 1);
	
    cudaMemcpy(h_dest_values, d_dest_values_2, sizeof(int)*output_length, cudaMemcpyDeviceToHost);	
	checkCudaErrors(cudaFree(d_source_values_2));
	checkCudaErrors(cudaFree(d_dest_values_2));

	delete[] h_source_values;
	delete[] h_dest_values;
}
