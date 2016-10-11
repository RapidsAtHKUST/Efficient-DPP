//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Foundation.h"
#include <cmath>
using namespace std;

int dataSize;           //max: MAX_DATA_SIZE
int outputSizeGS;      //output size of scatter and gather 

bool is_input;          //whether to read data or fast run
PlatInfo info;          //platform configuration structure

char input_arr_dir[500];
char input_rec_dir[500];
char input_loc_dir[500];

Record *fixedRecords;

int *fixedKeys;
int *fixedValues;

int *fixedLoc;

int experTime = 5;         //experiment time

#define NUM_FUNCS   (6)     //map, scatter, gather, reduce, scan, split
double bytes[NUM_FUNCS];

//for basic operation testing
#define MIN_BLOCK       (128)  
#define MAX_BLOCK       (1024)
#define MIN_GRID        (256)  
#define MAX_GRID        (32768)	
#define MAX_VEC_SIZE	(16)


#define NUM_BLOCK_VAR   (4)	
#define NUM_GRID_VAR    (8)	
#define NUM_VEC_SIZE    (5)

int vec[NUM_VEC_SIZE] = {1,2,4,8,16};
// int vec[NUM_VEC_SIZE] = {1};


//device basic operation performance matrix

//for vpu testing
Basic_info perfInfo_float[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];
Basic_info perfInfo_double[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];

//for all
Device_perf_info bestInfo;

template<typename T> void runVPU();
template<typename T> void runMem();
template<typename T> void runAccess();

void runBarrier(int experTime);
void runAtomic();
void runLatency();

double runMap();
void runGather();
void runScatter();
double runScan(int experTime, int& blockSize);
double runRadixSort(int experTime);

/*parameters:
 * if IS_INPUT==true,  executor INPUT_REC_DIR INPUT_ARR_DIR INPUT_LOC_DIR
 * else                executor DATASIZE
 * DATASIZE : data size
 * IS_INPUT : whether to input file from the file system
 * INPUT_REC_DIR : input directory of the record data if needed
 * INPUT_ARR_DIR : input directory of the array data if needed
 * INPUT_LOC_DIR : input directory of the location data if needed
 */
int main(int argc, const char * argv[]) {

    //platform initialization
    PlatInit* myPlatform = PlatInit::getInstance(0);
    cl_command_queue queue = myPlatform->getQueue();
    cl_context context = myPlatform->getContext();
    cl_command_queue currentQueue = queue;
    
    info.context = context;
    info.currentQueue = currentQueue;
    
    switch (argc) {
        case 2:         //fast run
            is_input = false;
            break;
        case 4:         //input
            is_input = true;
            break;
        default:
            cerr<<"Wrong number of parameters."<<endl;
            exit(1);
            break;
    }
    
    if (is_input) {
        strcat(input_rec_dir, argv[1]);
        strcat(input_arr_dir, argv[2]);
        strcat(input_loc_dir, argv[3]);
        std::cout<<"Start reading data..."<<std::endl;
        readFixedRecords(fixedRecords, input_rec_dir, dataSize);
        readFixedArray(fixedLoc, input_loc_dir, dataSize);
        std::cout<<"Finish reading data..."<<std::endl;
    }
    else {
        dataSize = atoi(argv[1]);
        // dataSize = 16000000;
        assert(dataSize > 0);
        outputSizeGS = 1000000;

    #ifdef RECORDS
        fixedKeys = new int[dataSize];
    #endif
        fixedValues = new int[dataSize];
    #ifdef RECORDS
        recordRandom<int>(fixedKeys, fixedValues, dataSize);
    #else
        valRandom<int>(fixedValues,dataSize, MAX_NUM);
    #endif
    }

    int map_blockSize = -1, map_gridSize = -1;
    int gather_blockSize = -1, gather_gridSize = -1;
    int scatter_blockSize = -1, scatter_gridSize = -1;
    int scan_blockSize = -1;

    int experTime = 1;
    double mapTime = 0.0f, gatherTime = 0.0f, scatterTime = 0.0f, scanTime = 0.0f, radixSortTime = 0.0f;

    // runVPU<int>();
    // runMem<int>();
    // runAccess<int>();
    // runMem<double>();
    // runAtomic();
    // runBarrier(experTime);
    // runLatency();

    // runMap();
    // runGather();
    // runScatter();
    scanTime = runScan(experTime, scan_blockSize);
    // radixSortTime = runRadixSort(experTime);

    cout<<"Time for scan: "<<scanTime<<" ms."<<'\t'<<"BlockSize: "<<scan_blockSize<<endl;
    // cout<<"Time for radix sort: "<<radixSortTime<<" ms."<<endl;
    
//    testSplit(fixedRecords, dataSize, info, 20, totalTime);           //fanout: 20
//    testBitonitSort(fixedRecords, dataSize, info, 1, totalTime);      //1:  ascendingls
//    testBitonitSort(fixedRecords, dataSize, info, 0, totalTime);      //0:  descending
    
//test joins
//    testNinlj(num1, num1, info, totalTime);
//    testInlj(num, num, info, totalTime);
//    testSmj(num, num, info, totalTime);
//    testHj(num, num, info, 16, totalTime);         //16: lower 16 bits to generate the buckets

    return 0;
}

template<typename T> 
void runVPU() {

    std::cout<<"----- Vector Instruction Throughput Test -----"<<std::endl;

    //block & grid limits
    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    assert(block_min >= block_min && block_max <= block_max &&
            grid_min >= grid_min && grid_max <= grid_max);

    int dataSize = block_max * grid_max; 
    std::cout<<"Max data size: "<<dataSize<<endl;
    assert(dataSize>0);

    Basic_info (*currentInfo)[NUM_GRID_VAR][NUM_VEC_SIZE];
    Basic_info currentBestInfo;

    char dataType[20];

    if (sizeof(T) == sizeof(int)) {
    	std::cout<<"Data type: int"<<std::endl;
    	strcpy(dataType, "int");
        currentInfo = perfInfo_float;
        currentBestInfo = bestInfo.float_info;
    }

    else if (sizeof(T) == sizeof(double))   {
    	std::cout<<"Data type: double"<<std::endl;
    	strcpy(dataType, "double");
        currentInfo = perfInfo_double;
        currentBestInfo = bestInfo.double_info;
    }
    else {
        std::cerr<<"Wrong data type!"<<std::endl;
        exit(1);
    }

    double bestTime = MAX_TIME;
    double bestThroughput = 0.0;
    double bestBlockSize = -1;
    double bestGridSize = -1;
    double bestVecSize = -1;

    T *input = (T*)malloc(sizeof(T)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = i / 2;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime, localThroughput;

                //--------test vpu------------
                testVPU<T>(input, info,localTime, blockSize, gridSize, vecSize);

                localThroughput = computeGFLOPS(blockSize * gridSize * vecSize, localTime, true, VPU_REPEAT_TIME,240);

                //print done!
                cout<<"blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<dataType<<vecSize<<'\t'<<localTime<<" ms"<<'\t'<<localThroughput<<" GFlops"<<"\tdone!"<<endl;

                //global update
                if (localThroughput > bestThroughput) {
                    bestThroughput = localThroughput;
                    bestTime = localTime;
                    bestBlockSize = blockSize;
                    bestGridSize = gridSize;
                    bestVecSize = vecSize;
                }

                //recording time and throughput
                currentInfo[blockIdx][gridIdx][vecIdx].vpu_time = localTime;
                currentInfo[blockIdx][gridIdx][vecIdx].vpu_throughput = localThroughput;
            }   
            gridIdx++;
        }
        blockIdx++;
    }

    //show information
    cout<<endl;
    cout<<"----------- Original ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# [vpu]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vecSize<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].vpu_time<<" ms"<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].vpu_throughput<<" GFlops"<<endl;
            }
            gridIdx++;
        }
        blockIdx++;
    }

    cout<<endl;
    cout<<"----------- Aggregated at vec_size ------------"<<endl;
    for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
        blockIdx = 0, gridIdx = 0;
        double bestTime = MAX_TIME;
        double bestThroughput = 0;
        double bestBlock = -1, bestGrid = -1;

        for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
            gridIdx = 0;
            for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].vpu_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].vpu_throughput;
                cout<<"# [vpu]:"
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GFlops"<<endl;
                if (throughput > bestThroughput) {
                    bestThroughput = throughput;
                    bestTime = elapsedTime;
                    bestBlock = blockSize;
                    bestGrid = gridSize;
                }
                gridIdx++;   
            }
            blockIdx++;
        }
        cout<<endl;
        cout<<"summary: "<<dataType<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GFlops\t"<<bestBlock<<'\t'<<bestGrid<<endl;
        cout<<"--------------------------------------------"<<endl;
    }

    cout<<endl;
    cout<<"----------- Aggregated at block & grid size ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {

            double bestTime = MAX_TIME;
            double bestThroughput = 0;
            int bestVecIdx = -1;

            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].vpu_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].vpu_throughput;
                
                //show information
                cout<<"# [vpu]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].vpu_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].vpu_throughput<<" GFlops"<<endl;

                if (throughput > bestThroughput) {
                    bestThroughput = throughput;
                    bestTime = elapsedTime;
                    bestVecIdx = vecIdx;
                }
            }
            cout<<endl;
            cout<<"summary: "
                <<blockSize<<'\t'
                <<gridSize<<'\t'
                <<bestTime<<" ms\t"
                <<bestThroughput<<" GFlops\t"
                <<dataType<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    currentBestInfo.vpu_time = bestTime;
    currentBestInfo.vpu_throughput = bestThroughput;
    currentBestInfo.vpu_blockSize = bestBlockSize;
    currentBestInfo.vpu_gridSize = bestGridSize;
    currentBestInfo.vpu_vecSize = bestVecSize;

    cout<<"Data type: "<<dataType<<endl;
    cout<<"Time for VPU: "<<currentBestInfo.vpu_time<<" ms."<<'\t'
        <<"BlockSize: "<<currentBestInfo.vpu_blockSize<<'\t'
        <<"GridSize: "<<currentBestInfo.vpu_gridSize<<'\t'
        <<"VecSize: "<<currentBestInfo.vpu_vecSize<<'\t'
        <<"Bandwidth: "<<currentBestInfo.vpu_throughput<<" GFLOPS"<<endl;
}

//when testing memory bandwidth, the dataSize should be sufficiently large, eg: 500M (2GB), larger than the LLC
template<typename T>
void runMem() {

    std::cout<<"-----  Memory Bandwidth Test ----- "<<std::endl;

    //this is the configuration for write and mul test
    //for read, shrink blockSize and gridSize by 2 respectively
    int blockSize = 1024, gridSize = 32768;
    int repeat_for_read = 100;

    Basic_info (*currentInfo)[NUM_GRID_VAR][NUM_VEC_SIZE];
    Basic_info currentBestInfo;

    char dataType[20];
    if (sizeof(T) == sizeof(int)) {
        std::cout<<"Data type: int"<<std::endl;
        strcpy(dataType, "int");
        currentInfo = perfInfo_float;
        currentBestInfo = bestInfo.float_info;
    }

    else if (sizeof(T) == sizeof(double))   {
        std::cout<<"Data type: double"<<std::endl;
        strcpy(dataType, "double");
        currentInfo = perfInfo_double;
        currentBestInfo = bestInfo.double_info;
    }
    else {
        std::cerr<<"Wrong data type!"<<std::endl;
        exit(1);
    }

    double bestTime_read = MAX_TIME;
    double throughput_read = 0.0;

    double bestTime_write = MAX_TIME;
    double throughput_write = 0.0;

    double bestTime_mul = MAX_TIME;
    double throughput_mul = 0.0;

    double bestTime_add = MAX_TIME;
    double throughput_add = 0.0;

    //run the memory test
    testMem<T>(info, blockSize, gridSize, bestTime_read, bestTime_write, bestTime_mul, bestTime_add , repeat_for_read);
    //compute the throughput
    throughput_read = computeMem(blockSize*gridSize/4, sizeof(T), bestTime_read);
    throughput_write = computeMem(blockSize*gridSize, sizeof(T), bestTime_write);
    throughput_mul = computeMem(blockSize*gridSize*2*2, sizeof(T), bestTime_mul);
    throughput_add = computeMem(blockSize*gridSize*2*3, sizeof(T), bestTime_add);

    currentBestInfo.mem_read_time = bestTime_read;
    currentBestInfo.mem_read_throughput = throughput_read;

    currentBestInfo.mem_write_time = bestTime_write;
    currentBestInfo.mem_write_throughput = throughput_write;
    
    currentBestInfo.mem_mul_time = bestTime_mul;
    currentBestInfo.mem_mul_throughput = throughput_mul;

    currentBestInfo.mem_add_time = bestTime_add;
    currentBestInfo.mem_add_throughput = throughput_add;

    cout<<"Time for memory read(Repeat:"<<repeat_for_read<<"): "<<currentBestInfo.mem_read_time<<" ms."<<'\t'
        <<"Bandwidth: "<<currentBestInfo.mem_read_throughput<<" GB/s"<<endl;

    cout<<"Time for memory write: "<<currentBestInfo.mem_write_time<<" ms."<<'\t'
        <<"Bandwidth: "<<currentBestInfo.mem_write_throughput<<" GB/s"<<endl;

    cout<<"Time for memory mul: "<<currentBestInfo.mem_mul_time<<" ms."<<'\t'
    <<"Bandwidth: "<<currentBestInfo.mem_mul_throughput<<" GB/s"<<endl;

    cout<<"Time for memory add: "<<currentBestInfo.mem_add_time<<" ms."<<'\t'
    <<"Bandwidth: "<<currentBestInfo.mem_add_throughput<<" GB/s"<<endl;
}

template<typename T>
void runAccess() {

    std::cout<<"-----  Memory Access Test ----- "<<std::endl;

    int blockSize = 512, gridSize = 16384;
    int access_repeat = 100;

    char dataType[20];
    if (sizeof(T) == sizeof(int)) {
        std::cout<<"Data type: int"<<std::endl;
    }

    else if (sizeof(T) == sizeof(double))   {
        std::cout<<"Data type: double"<<std::endl;
    }
    else {
        std::cerr<<"Wrong data type!"<<std::endl;
        exit(1);
    }

    //run the memory access test
    testAccess<T>(info, blockSize, gridSize, access_repeat);
}

void runBarrier(int experTime) {

    cout<<"----------- Barrier test ------------"<<endl;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                //--------test map------------
                double percentage;
                testBarrier(                
                input, info,tempTime, percentage, blockSize, gridSize);

                cout<<"blockSize: "<<blockSize<<'\t'
            	<<"gridSize: "<<gridSize<<'\t'
            	<<"barrier time: "<<tempTime<<" ms\t"
            	<<"time per thread: "<<tempTime/(blockSize *  gridSize)*1e6<<" ns\t"
            	<<"percentage: "<<percentage <<"%"<<endl;
            }
        }
    }

    delete[] input;
}

void runAtomic() {

	std::cout<<"----- Local Atomic Test -----"<<std::endl;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    // int blockIdx = 0, gridIdx = 0;
    // for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
    // 	gridIdx = 0;
    //     for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
    //         	double localTime;

    //             //--------test local atomic------------
    //             testAtomic(                
    //           	info,localTime, blockSize, gridSize, true);

    //         gridIdx++;
    //     }
    //     blockIdx++;
    // }

    for(int gz = 1; gz <= 1024; gz++) {
        for(int r = 1; r <=3; r++) {
            double localTime;
            testAtomic(info,localTime, 512, gz, false);
        }
        
    }

    // std::cout<<"----- Global Atomic Test -----"<<std::endl;

    // blockIdx = 0, gridIdx = 0;
    // for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
    //     gridIdx = 0;
    //     for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
    //         	double localTime;

    //             //--------test global atomic------------
    //             testAtomic(                
    //           	info,localTime, blockSize, gridSize, false);

    //         gridIdx++;
    //     }
    //     blockIdx++;
    // }
}

void runLatency() {
    testLatency(info);
}

double runMap() {
    int blockSize = 1024, gridSize = 8192;
    int repeat = 64, repeat_trans = 16;
    testMap(info, repeat, repeat_trans, blockSize, gridSize);
}

void runGather() {
    cout<<"---------- Gather test ---------"<<endl;

    double totalTime;
    bool res;

    int blockSize = 1024, gridSize = 2048;

    int run = 7, dataBegin = 0, dataEnd = 19;
    
    //multi-pass : 7 choices
    int numOfRun[7] = {1,2,4,8,16,32,64};
    int myDataSize[19] = {250000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 100000000,200000000,300000000,400000000,500000000,600000000,700000000,800000000,900000000,1000000000};

    //warm up
    testGather(            
    #ifdef RECORDS
            fixedKeys,
    #endif
            fixedValues,
            1000000, 1000000, info , numOfRun, run, NULL, false, blockSize, gridSize);
    
    cout<<"---------- Warm up over, begin test ---------"<<endl;
    
    double gatherTime[19][7] = {0.0};

    for(int idx = dataBegin; idx < dataEnd; idx++) {
        //input size should be larger or equals to the output size
        int inputSize = myDataSize[idx];
        int outputSize = myDataSize[idx];   //for equal case
        // int outputSize = outputSizeGS;   //for 1M case

        res = testGather(            
    #ifdef RECORDS
        fixedKeys,
    #endif
        fixedValues, 
        inputSize, outputSize, info , numOfRun, run, gatherTime[idx], true, blockSize, gridSize);

        //time per tuple and change from "ms" to "ns"
        for(int cRun = 0; cRun < run; cRun++) {
            gatherTime[idx][cRun] = gatherTime[idx][cRun] / outputSize * 1e6;  
        }
    }

    cout<<"------------------------------"<<endl;
    cout<<"Gather stat:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"Current # of pass:"<<numOfRun[r]<<endl;
        for(int i = dataBegin; i < dataEnd; i++) {
            cout<<"Data size: "<<myDataSize[i]<<'\t'<<"time per tuple: "<<gatherTime[i][r]<<" ns."<<endl;
        }
    }
    cout<<endl;

    cout<<"For python:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"gather_"<<numOfRun[r]<<" = ";
        cout<<"["<<gatherTime[dataBegin][r];
        for(int i = dataBegin+1; i < dataEnd; i++) {
            cout<<','<<gatherTime[i][r];
        }
        cout<<"]"<<endl;
    }
    
    cout<<"For excel:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"Current # of pass:"<<numOfRun[r]<<endl;
        for(int i = dataBegin; i < dataEnd; i++) {
            cout<<gatherTime[i][r]<<endl;
        }
        cout<<endl;
    }
}

void runScatter() {
    cout<<"---------- Scatter test ---------"<<endl;
    double totalTime;
    bool res;

    int blockSize = 1024, gridSize = 2048;
    int run = 7, dataBegin = 0, dataEnd = 19;

    int numOfRun[7] = {1,2,4,8,16,32,64};
    int myDataSize[19] = {250000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 100000000,200000000,300000000,400000000,500000000,600000000,700000000,800000000,900000000,1000000000};

    //warm up
    testScatter(            
    #ifdef RECORDS
            fixedKeys,
    #endif
            fixedValues,
            1000000,1000000, info , numOfRun, run, NULL, false, blockSize, gridSize);
    
    cout<<"---------- Warm up over, begin test ---------"<<endl;
    
    // --------test scatter------------
    double scatterTime[19][7] = {0.0};

    for(int idx = dataBegin; idx < dataEnd; idx++) {
        //input size should be smaller or equals to the output size
        
        // int inputSize = outputSizeGS;   //for 1M size case
        int inputSize = myDataSize[idx];    //for equal size case
        int outputSize = myDataSize[idx];

        res = testScatter(            
    #ifdef RECORDS
        fixedKeys,
    #endif
        fixedValues, 
        inputSize, outputSize , info , numOfRun, run, scatterTime[idx], true, blockSize, gridSize);

        //time per tuple and change from "ms" to "ns"
        for(int cRun = 0; cRun < run; cRun++) {
            scatterTime[idx][cRun] = scatterTime[idx][cRun] / inputSize * 1e6;  
        }
    }
    
    cout<<"------------------------------"<<endl;
    cout<<"Scatter stat:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"Current # of pass:"<<numOfRun[r]<<endl;
        for(int i = dataBegin; i < dataEnd; i++) {
            cout<<"Data size: "<<myDataSize[i]<<'\t'<<"time per tuple: "<<scatterTime[i][r]<<" ns."<<endl;
        }
    }
    cout<<endl;

    cout<<"For python:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"scatter_"<<numOfRun[r]<<" = ";
        cout<<"["<<scatterTime[dataBegin][r];
        for(int i = dataBegin+1; i < dataEnd; i++) {
            cout<<','<<scatterTime[i][r];
        }
        cout<<"]"<<endl;
    }
    
    cout<<"For excel:"<<endl;
    for(int r = 0; r < run; r++) {
        cout<<"Current # of pass:"<<numOfRun[r]<<endl;
        for(int i = dataBegin; i < dataEnd; i++) {
            cout<<scatterTime[i][r]<<endl;
        }
        cout<<endl;
    }
}

double runScan(int experTime, int& bestBlockSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bool res;

    int block_min = 1024, block_max = 1024;
    int grid_min = 1024, grid_max = 1024;
    
    cout<<"--------- Exclusive --------"<<endl;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {   
        double tempTime = MAX_TIME;
        for(int i = 0 ; i < experTime; i++) {       
            // --------test scan------------
            res = testScan(fixedValues, dataSize, info, tempTime, 1, blockSize);             //0: inclusive
        }
        if (tempTime < bestTime && res == true) {
            bestTime = tempTime;
            bestBlockSize = blockSize;
        }
    }

    cout<<"--------- Inclusive --------"<<endl;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {   
        double tempTime = MAX_TIME;
        for(int i = 0 ; i < experTime; i++) {       
            // --------test scan------------
            res = testScan(fixedValues, dataSize, info, tempTime, 0, blockSize);             //0: inclusive
        }
        if (tempTime < bestTime && res == true) {
            bestTime = tempTime;
            bestBlockSize = blockSize;
        }
    }
    return bestTime;
}

//no need to set blockSize and gridSize
double runRadixSort(int experTime) {
    double bestTime = MAX_TIME;
    bool res;
    
    double tempTime = MAX_TIME;
    for(int i = 0 ; i < experTime; i++) {       
        //--------test radix sort------------
        res = testRadixSort(          
#ifdef RECORDS
        fixedKeys,
#endif
        fixedValues, 
        dataSize, info, tempTime);
    }
    if (tempTime < bestTime && res == true) {
        bestTime = tempTime;
    }
    return bestTime;
}