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
bool is_input;          //whether to read data or fast run
PlatInfo info;          //platform configuration structure

char input_arr_dir[500];
char input_rec_dir[500];
char input_loc_dir[500];

Record *fixedRecords;

int *fixedKeys;
int *fixedValues;

int *fixedLoc;

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
#define NUM_VEC_SIZE    (1)

// int vec[NUM_VEC_SIZE] = {1,2,4,8,16};
int vec[NUM_VEC_SIZE] = {1};


//device basic operation performance matrix
Basic_info perfInfo_float[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];
Basic_info perfInfo_double[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];
Device_perf_info bestInfo;

template<typename T> void runVPU();
template<typename T> void runMemReadWrite();
template<typename T> void runMemTriad();

void runBarrier(int experTime);
void runAtomic();

double runMap(int expeTime, int& blockSize, int& gridSize);
double runGather(int experTime, int& blockSize, int& gridSize);
double runScatter(int experTime, int& blockSize, int& gridSize);
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

    // #ifdef RECORDS
    //     fixedKeys = new int[dataSize];
    // #endif
    //     fixedValues = new int[dataSize];
    //     fixedLoc = new int[dataSize];
    // #ifdef RECORDS
    //     recordRandom<int>(fixedKeys, fixedValues, dataSize);
    // #else
    //     valRandom<int>(fixedValues,dataSize, MAX_NUM);
    // #endif
    //     valRandom_Only<int>(fixedLoc, dataSize, SHUFFLE_TIME(dataSize));
    }

    int map_blockSize = -1, map_gridSize = -1;
    int gather_blockSize = -1, gather_gridSize = -1;
    int scatter_blockSize = -1, scatter_gridSize = -1;
    int scan_blockSize = -1;

    int experTime = 1;
    double mapTime = 0.0f, gatherTime = 0.0f, scatterTime = 0.0f, scanTime = 0.0f, radixSortTime = 0.0f;

	// runBarrier(experTime);

    // runVPU<float>(experTime, vpu_blockSize, vpu_gridSize, vpu_vecSize, bestInfo);
    // runMemRead<float>(experTime, mem_read_blockSize, mem_read_gridSize, mem_read_vecSize, bestInfo);
    // runMemWrite<float>(experTime, mem_write_blockSize, mem_write_gridSize, mem_write_vecSize, bestInfo);

    // runMemTriad<float>(experTime, mem_triad_blockSize, mem_triad_gridSize, mem_triad_vecSize, bestInfo);

    // mapTime = runMap(experTime, map_blockSize, map_gridSize);
    // gatherTime = runGather(experTime, gather_blockSize, gather_gridSize);
    // scatterTime = runScatter(experTime, scatter_blockSize, scatter_gridSize);
    // scanTime = runScan(experTime, scan_blockSize);
    // radixSortTime = runRadixSort(experTime);

	// runAtomic();

//bandwidth calculation
    // bytes[0] = dataSize * sizeof(int) * 2;

    // cout<<"Time for map: "<<mapTime<<" ms."<<'\t'
    //     <<"BlockSize: "<<map_blockSize<<'\t'
    //     <<"GridSize: "<<map_gridSize<<'\t'
    //     <<"Bandwidth:"<<1.0E-06 * bytes[0] / mapTime<<" GB/s." 
    //     <<endl;
    
    // cout<<"Time for gather: "<<gatherTime<<" ms."<<'\t'<<"BlockSize: "<<gather_blockSize<<'\t'<<"GridSize: "<<gather_gridSize<<endl;
    // cout<<"Time for scatter: "<<scatterTime<<" ms."<<'\t'<<"BlockSize: "<<scatter_blockSize<<'\t'<<"GridSize: "<<scatter_gridSize<<endl;
    // cout<<"Time for scan: "<<scanTime<<" ms."<<'\t'<<"BlockSize: "<<scan_blockSize<<endl;
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
    assert(dataSize>0);

    Basic_info*** currentInfo;
    Basic_info currentBestInfo;

    char dataType[20];

    if (sizeof(T) == sizeof(float)) {
    	std::cout<<"Data type: float"<<std::endl;
    	strcpy(dataType, "float");
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
        input[i] = 0;
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
                cout<<"# "
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
                cout<<"# "
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
                cout<<"# "
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
        <<"Bandwidth: "<<currentBestInfo.vpu_throughput<<"GFLOPS"<<endl;
}

//when testing memory bandwidth, the dataSize should be sufficiently large, eg: 500M (2GB), larger than the LLC
template<typename T>
void runMemReadWrite() {

    std::cout<<"-----  Memory Bandwidth Read Write Test ----- "<<std::endl;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;
    
    assert(block_min >= block_min && block_max <= block_max &&
            grid_min >= grid_min && grid_max <= grid_max);


    int dataSize = block_max * grid_max * MAX_VEC_SIZE;
    assert(dataSize>0);

    Basic_info*** currentInfo;
    Basic_info currentBestInfo;

    char dataType[20];

    if (sizeof(T) == sizeof(float)) {
        std::cout<<"Data type: float"<<std::endl;
        strcpy(dataType, "float");
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
    double bestThroughput_read = 0.0;
    double bestBlockSize_read = -1;
    double bestGridSize_read = -1;
    double bestVecSize_read = -1;

    double bestTime_write = MAX_TIME;
    double bestThroughput_write = 0.0;
    double bestBlockSize_write = -1;
    double bestGridSize_write = -1;
    double bestVecSize_write = -1;

    T *input = (T*)malloc(sizeof(T)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 1.7682;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime_read = 0;
                double localThroughput_read = 0;

                double localTime_write = 0;
                double localThroughput_write = 0;

                testMemReadWrite<T>(input, info,localTime_read, localTime_write, blockSize, gridSize, vecSize);

                localThroughput_read = computeMem(blockSize*gridSize*vecSize, sizeof(T), localTime_read);
                localThroughput_write = computeMem(blockSize*gridSize*vecSize, sizeof(T), localTime_write);

                //print done!
                cout<<"read: blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<dataType<<vecSize<<'\t'<<localTime_read<<" ms\t"<<localThroughput_read<<" GB/s\tdone!"<<endl;

                cout<<"write: blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<dataType<<vecSize<<'\t'<<localTime_write<<" ms\t"<<localThroughput_write<<" GB/s\tdone!"<<endl;

                //global update
                if (localThroughput_read > bestThroughput_read) {
                    bestTime_read = localTime_read;
                    bestThroughput_read = localThroughput_read;
                    bestBlockSize_read = blockSize;
                    bestGridSize_read = gridSize;
                    bestVecSize_read = vecSize;
                }
                if (localThroughput_write > bestThroughput_write) {
                    bestTime_write = localTime_write;
                    bestThroughput_write = localThroughput_write;
                    bestBlockSize_write = blockSize;
                    bestGridSize_write = gridSize;
                    bestVecSize_write = vecSize;
                }

                //recording time and throughput
                currentInfo[blockIdx][gridIdx][vecIdx].mem_read_time = localTime_read;
                currentInfo[blockIdx][gridIdx][vecIdx].mem_read_throughput = localThroughput_read;

                currentInfo[blockIdx][gridIdx][vecIdx].mem_write_time = localTime_write;
                currentInfo[blockIdx][gridIdx][vecIdx].mem_write_throughput = localThroughput_write;
            }   
            gridIdx++;
        }
        blockIdx++;
    }

    //show read information
    cout<<endl;
    cout<<"----------------------------  Read Info -----------------------------"<<endl<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# [read]: "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vecSize<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_read_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_read_throughput<<" GB/s"<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_read_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_read_throughput;
                cout<<"# [read]: "
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GB/s"<<endl;
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
        cout<<"summary: "<<dataType<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GB/s\t"<<bestBlock<<'\t'<<bestGrid<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_read_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_read_throughput;
                
                //show information
                cout<<"# [read]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_read_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_read_throughput<<" GB/s"<<endl;

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
                <<bestThroughput<<" GB/s\t"
                <<dataType<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }


    //show write information
    cout<<endl;
    cout<<"----------------------------  Write Info -----------------------------"<<endl<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# [write]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vecSize<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_write_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_write_throughput<<" GB/s"<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_write_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_write_throughput;
                cout<<"# [write]:"
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GB/s"<<endl;
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
        cout<<"summary: "<<dataType<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GB/s\t"<<bestBlock<<'\t'<<bestGrid<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_write_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_write_throughput;
                
                //show information
                cout<<"# [write]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_write_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_write_throughput<<" GB/s"<<endl;

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
                <<bestThroughput<<" GB/s\t"
                <<dataType<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    currentBestInfo.mem_read_time = bestTime_read;
    currentBestInfo.mem_read_throughput = bestThroughput_read;
    currentBestInfo.mem_read_blockSize = bestBlockSize_read;
    currentBestInfo.mem_read_gridSize = bestGridSize_read;
    currentBestInfo.mem_read_vecSize = bestVecSize_read;

    currentBestInfo.mem_write_time = bestTime_write;
    currentBestInfo.mem_write_throughput = bestThroughput_write;
    currentBestInfo.mem_write_blockSize = bestBlockSize_write;
    currentBestInfo.mem_write_gridSize = bestGridSize_write;
    currentBestInfo.mem_write_vecSize = bestVecSize_write;

    cout<<"Data type: "<<dataType<<endl;
    cout<<"Time for memory read: "<<currentBestInfo.mem_read_time<<" ms."<<'\t'
        <<"BlockSize: "<<currentBestInfo.mem_read_blockSize<<'\t'
        <<"GridSize: "<<currentBestInfo.mem_read_gridSize<<'\t'
        <<"VecSize: "<<currentBestInfo.mem_read_vecSize<<'\t'
        <<"Bandwidth: "<<currentBestInfo.mem_read_throughput<<" GB/s"<<endl;

    cout<<"Time for memory write: "<<currentBestInfo.mem_write_time<<" ms."<<'\t'
        <<"BlockSize: "<<currentBestInfo.mem_write_blockSize<<'\t'
        <<"GridSize: "<<currentBestInfo.mem_write_gridSize<<'\t'
        <<"VecSize: "<<currentBestInfo.mem_read_vecSize<<'\t'
        <<"Bandwidth: "<<currentBestInfo.mem_write_throughput<<" GB/s"<<endl;
}

template<typename T>
void runMemTriad() {

    std::cout<<"-----  Memory Bandwidth Triad Test ----- "<<std::endl;

    Basic_info*** currentInfo;
    Basic_info currentBestInfo;

    char dataType[20];

    if (sizeof(T) == sizeof(float)) {
        std::cout<<"Data type: float"<<std::endl;
        strcpy(dataType, "float");
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

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;
    
    assert(block_min >= block_min && block_max <= block_max &&
            grid_min >= grid_min && grid_max <= grid_max);

    int dataSize = block_max * grid_max * MAX_VEC_SIZE;
    assert(dataSize>0);

    double bestTime = MAX_TIME;
    double bestThroughput = 0.0;
    double bestBlockSize = -1;
    double bestGridSize = -1;
    double bestVecSize = -1;

    T *input = (T*)malloc(sizeof(T)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 1.54;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime = 0;
                double localThroughput = 0;

                //--------test memery triad------------
                testTriad<T>(input, info,localTime, blockSize, gridSize, vecSize);

                localThroughput = computeMem(blockSize * gridSize * vecSize * 3, sizeof(T), localTime);

                //print done!
                cout<<"blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<dataType<<vecSize<<'\t'<<localTime<<" ms\t"<<localThroughput<<" GB/s\tdone!"<<endl;

                //global update
                if (localThroughput > bestThroughput) {
                    bestTime = localTime;
                    bestThroughput = localThroughput;
                    bestBlockSize = blockSize;
                    bestGridSize = gridSize;
                    bestVecSize = vecSize;
                }

                //recording time and throughput
                currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_time = localTime;
                currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_throughput = localThroughput;
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
                cout<<"# [triad]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vecSize<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_throughput<<" GB/s"<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_throughput;
                cout<<"# [triad]:"
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GB/s"<<endl;
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
        cout<<"summary: "<<dataType<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GB/s\t"<<bestBlock<<'\t'<<bestGrid<<endl;
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
                double elapsedTime = currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_time;
                double throughput = currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_throughput;
                
                //show information
                cout<<"# [triad]:"
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<dataType<<vec[vecIdx]<<'\t'
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_time<<" ms\t"
                    <<currentInfo[blockIdx][gridIdx][vecIdx].mem_triad_throughput<<" GB/s"<<endl;

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
                <<bestThroughput<<" GB/s\t"
                <<dataType<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    currentBestInfo.mem_triad_time = bestTime;
    currentBestInfo.mem_triad_throughput = bestThroughput;
    currentBestInfo.mem_triad_blockSize = bestBlockSize;
    currentBestInfo.mem_triad_gridSize = bestGridSize;
    currentBestInfo.mem_triad_vecSize= bestVecSize;

    cout<<"Data type: "<<dataType<<endl;
    cout<<"Time for memory triad: "<<currentBestInfo.mem_triad_time<<" ms."<<'\t'
    	<<"BlockSize: "<<currentBestInfo.mem_triad_blockSize<<'\t'
    	<<"GridSize: "<<currentBestInfo.mem_triad_gridSize<<'\t'
        <<"VecSize： "<<currentBestInfo.mem_triad_vecSize<<'\t'
    	<<"Bandwidth: "<<currentBestInfo.mem_triad_throughput<<" GB/s"<<endl;
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
double runMap(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

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
                res = testMap(                
#ifdef RECORDS
                fixedKeys,
#endif
                input,
                dataSize,info,tempTime, blockSize, gridSize);
            }
            // if (tempTime < bestTime && res == true) {
            if (tempTime < bestTime) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }

    delete[] input;

    return bestTime;
}

double runGather(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                // --------test gather------------
                res = testGather(            
#ifdef RECORDS
                fixedKeys,
#endif
                fixedValues, 
                dataSize,  info , tempTime, blockSize, gridSize);
            }
            if (tempTime < bestTime && res == true) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }

    return bestTime;
}

double runScatter(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;

    for(int blockSize = block_min; blockSize <= block_max; blockSize<<=1) {
        for(int gridSize = grid_min; gridSize <= grid_max; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {  
                // --------test scatter------------     
                res = testScatter(             
#ifdef RECORDS
                fixedKeys,
#endif
                fixedValues, 
                dataSize,  info , tempTime, blockSize, gridSize);
            }
            if (tempTime < bestTime && res == true) {
                bestTime = tempTime;
                bestBlockSize = blockSize;
                bestGridSize = gridSize;
            }
        }
    }

    return bestTime;
}

double runScan(int experTime, int& bestBlockSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bool res;

    int block_min = 128, block_max = 1024;
    int grid_min = 256, grid_max = 32768;
     
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