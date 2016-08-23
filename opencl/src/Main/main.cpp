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
#define MIN_BLOCK   (128)  //128
#define MAX_BLOCK   (1024)
#define MIN_GRID    (256)  //256
#define MAX_GRID    (32768)

#define NUM_BLOCK_VAR   (4)
#define NUM_GRID_VAR    (8)
#define NUM_VEC_SIZE    (5)

int vec[NUM_VEC_SIZE] = {1,2,4,8,16};
// int vec[NUM_VEC_SIZE] = {1};


//device basic operation performance matrix
Device_perf_info perfInfo[NUM_BLOCK_VAR][NUM_GRID_VAR][NUM_VEC_SIZE];
Device_perf_info bestInfo;

void runVPU(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo);
void  runMemRead(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo);
void runMemWrite(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo);
void runBarrier(int experTime);

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
    
    int vpu_vec_size = -1;  //for VPU test
    int mem_read_vec_size = -1;
    int mem_write_vec_size = -1;

    int vpu_blockSize = -1, vpu_gridSize = -1;
    int mem_read_blockSize = -1, mem_read_gridSize = -1;
    int mem_write_blockSize = -1, mem_write_gridSize = -1;

    int map_blockSize = -1, map_gridSize = -1;
    int gather_blockSize = -1, gather_gridSize = -1;
    int scatter_blockSize = -1, scatter_gridSize = -1;
    int scan_blockSize = -1;

    int experTime = 1;
    double vpuTime = 0.0f, memReadTime = 0.0f, mapTime = 0.0f, gatherTime = 0.0f, scatterTime = 0.0f, scanTime = 0.0f, radixSortTime = 0.0f;

	runBarrier(experTime);

    // runVPU(experTime, vpu_blockSize, vpu_gridSize, vpu_vec_size, bestInfo);
    // runMemRead(experTime, mem_read_blockSize, mem_read_gridSize, mem_read_vec_size, bestInfo);
    // runMemWrite(experTime, mem_write_blockSize, mem_write_gridSize, mem_write_vec_size, bestInfo);

    // mapTime = runMap(experTime, map_blockSize, map_gridSize);
    // gatherTime = runGather(experTime, gather_blockSize, gather_gridSize);
    // scatterTime = runScatter(experTime, scatter_blockSize, scatter_gridSize);
    // scanTime = runScan(experTime, scan_blockSize);
    // radixSortTime = runRadixSort(experTime);

//bandwidth calculation
    bytes[0] = dataSize * sizeof(int) * 2;

    // cout<<"Time for map: "<<mapTime<<" ms."<<'\t'
    //     <<"BlockSize: "<<map_blockSize<<'\t'
    //     <<"GridSize: "<<map_gridSize<<'\t'
    //     <<"Bandwidth:"<<1.0E-06 * bytes[0] / mapTime<<" GB/s." 
    //     <<endl;
    

    // cout<<"Time for VPU: "<<vpuTime<<" ms."<<'\t'
    //     <<"BlockSize: "<<vpu_blockSize<<'\t'
    //     <<"GridSize: "<<vpu_gridSize<<'\t'
    //     <<"VecSize: "<<vpu_vec_size<<'\t'
    //     <<"Bandwidth: "<<computeGFLOPS(dataSize, vpuTime, true, VPU_REPEAT_TIME,240)<<"GFLOPS"<<endl;

    // cout<<"Time for memory read: "<<bestInfo.float_info.mem_read_time<<" ms."<<'\t'
    //     <<"BlockSize: "<<mem_read_blockSize<<'\t'
    //     <<"GridSize: "<<mem_read_gridSize<<'\t'
    //     <<"Bandwidth: "<<bestInfo.float_info.mem_read_throughput<<" GB/s"<<endl;

    // cout<<"Time for memory write: "<<bestInfo.float_info.mem_write_time<<" ms."<<'\t'
    //     <<"BlockSize: "<<mem_write_blockSize<<'\t'
    //     <<"GridSize: "<<mem_write_gridSize<<'\t'
    //     <<"Bandwidth: "<<bestInfo.float_info.mem_write_throughput<<" GB/s"<<endl;


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

void runVPU(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo) {

    std::cout<<"----- Vector Instruction Throughput Test -----"<<std::endl;

    double bestTime = MAX_TIME;
    double bestThroughput = 0.0;
    bestBlockSize = -1;
    bestGridSize = -1;
    bestVecSize = -1;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime = 0;
                double localThroughput = 0;

                //--------test vpu------------
                testVPU(                
                input,
                dataSize,info,localTime, blockSize, gridSize, vecSize);

                localThroughput = computeGFLOPS(dataSize, localTime, true, VPU_REPEAT_TIME,240);

                //print done!
                cout<<"blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<"float"<<vecSize<<'\t'<<localTime<<" ms"<<'\t'<<localThroughput<<" GFlops"<<"\tdone!"<<endl;

                //global update
                if (localTime < bestTime) {
                    bestTime = localTime;
                    bestThroughput = localThroughput;
                    bestBlockSize = blockSize;
                    bestGridSize = gridSize;
                    bestVecSize = vecSize;
                }

                //recording time and throughput
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_time = localTime;
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_throughput = localThroughput;
            }   
            gridIdx++;
        }
        blockIdx++;
    }

    //show information
    cout<<endl;
    cout<<"----------- Original ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vecSize<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_time<<" ms"<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_throughput<<" GFlops"<<endl;
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

        for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
            gridIdx = 0;
            for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_throughput;
                cout<<"# "
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GFlops"<<endl;
                if (elapsedTime < bestTime) {
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
        cout<<"summary: "<<"float"<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GFlops\t"<<bestBlock<<'\t'<<bestGrid<<endl;
        cout<<"--------------------------------------------"<<endl;
    }

    cout<<endl;
    cout<<"----------- Aggregated at block & grid size ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {

            double bestTime = MAX_TIME;
            double bestThroughput = 0;
            int bestVecIdx = -1;

            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_throughput;
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_time<<" ms\t"
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.vpu_throughput<<" GFlops"<<endl;

                if (elapsedTime < bestTime) {
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
                <<"float"<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    bestInfo.float_info.vpu_time = bestTime;
    bestInfo.float_info.vpu_throughput = bestThroughput;
}

//when testing memory bandwidth, the dataSize should be sufficiently large, eg: 500M (2GB), larger than the LLC
void runMemRead(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo) {

    std::cout<<"-----  Memory Bandwidth Read Test ----- "<<std::endl;

    double bestTime = MAX_TIME;
    double bestThroughput = 0.0;
    bestBlockSize = -1;
    bestGridSize = -1;
    bestVecSize = -1;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 1.7682;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime = 0;
                double localThroughput = 0;

                testMemRead(                
                input,
                dataSize,info,localTime, blockSize, gridSize, vecSize);
                    
                localThroughput = computeMem(dataSize, sizeof(float), localTime);

                //print done!
                cout<<"blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<"float"<<vecSize<<'\t'<<localTime<<" ms\t"<<localThroughput<<" GB/s\tdone!"<<endl;

                //global update
                if (localTime < bestTime) {
                    bestTime = localTime;
                    bestThroughput = localThroughput;
                    bestBlockSize = blockSize;
                    bestGridSize = gridSize;
                    bestVecSize = vecSize;
                }

                //recording time and throughput
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_time = localTime;
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_throughput = localThroughput;
            }   
            gridIdx++;
        }
        blockIdx++;
    }

    //show information
    cout<<endl;
    cout<<"----------- Original ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vecSize<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_time<<" ms\t"
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_throughput<<" GB/s"<<endl;
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

        for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
            gridIdx = 0;
            for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_throughput;
                cout<<"# "
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GB/s"<<endl;
                if (elapsedTime < bestTime) {
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
        cout<<"summary: "<<"float"<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GB/s\t"<<bestBlock<<'\t'<<bestGrid<<endl;
        cout<<"--------------------------------------------"<<endl;
    }

    cout<<endl;
    cout<<"----------- Aggregated at block & grid size ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {

            double bestTime = MAX_TIME;
            double bestThroughput = 0;
            int bestVecIdx = -1;

            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_throughput;
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_time<<" ms\t"
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_read_throughput<<" GB/s"<<endl;

                if (elapsedTime < bestTime) {
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
                <<"float"<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    bestInfo.float_info.mem_read_time = bestTime;
    bestInfo.float_info.mem_read_throughput = bestThroughput;
}

void runMemWrite(int experTime, int& bestBlockSize, int& bestGridSize, int& bestVecSize, Device_perf_info& bestInfo) {

    std::cout<<"-----  Memory Bandwidth Write Test ----- "<<std::endl;

    double bestTime = MAX_TIME;
    double bestThroughput = 0.0;
    bestBlockSize = -1;
    bestGridSize = -1;
    bestVecSize = -1;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    int blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                double localTime = 0;
                double localThroughput = 0;

                //--------test memery write------------
                testMemWrite(                
                dataSize,info,localTime, blockSize, gridSize, vecSize);

                localThroughput = computeMem(dataSize, sizeof(float), localTime);

                //print done!
                cout<<"blockSize="<<blockSize<<'\t'<<"gridSize="<<gridSize<<'\t'<<"float"<<vecSize<<'\t'<<localTime<<" ms\t"<<localThroughput<<" GB/s\tdone!"<<endl;

                //global update
                if (localTime < bestTime) {
                    bestTime = localTime;
                    bestThroughput = localThroughput;
                    bestBlockSize = blockSize;
                    bestGridSize = gridSize;
                    bestVecSize = vecSize;
                }

                //recording time and throughput
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_time = localTime;
                perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_throughput = localThroughput;
            }   
            gridIdx++;
        }
        blockIdx++;
    }

    //show information
    cout<<endl;
    cout<<"----------- Original ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                int vecSize = vec[vecIdx];
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vecSize<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_time<<" ms\t"
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_throughput<<" GB/s"<<endl;
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

        for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
            gridIdx = 0;
            for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_throughput;
                cout<<"# "
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<elapsedTime<<" ms\t"
                    <<throughput<<" GB/s"<<endl;
                if (elapsedTime < bestTime) {
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
        cout<<"summary: "<<"float"<<vec[vecIdx]<<'\t'<<bestTime<<" ms\t"<<bestThroughput<<" GB/s\t"<<bestBlock<<'\t'<<bestGrid<<endl;
        cout<<"--------------------------------------------"<<endl;
    }

    cout<<endl;
    cout<<"----------- Aggregated at block & grid size ------------"<<endl;
    blockIdx = 0, gridIdx = 0;
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        gridIdx = 0;
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {

            double bestTime = MAX_TIME;
            double bestThroughput = 0;
            int bestVecIdx = -1;

            for(int vecIdx = 0; vecIdx < NUM_VEC_SIZE; vecIdx++) {
                double elapsedTime = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_time;
                double throughput = perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_throughput;
                
                //show information
                cout<<"# "
                    <<blockSize<<'\t'
                    <<gridSize<<'\t'
                    <<"float"<<vec[vecIdx]<<'\t'
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_time<<" ms\t"
                    <<perfInfo[blockIdx][gridIdx][vecIdx].float_info.mem_write_throughput<<" GB/s"<<endl;

                if (elapsedTime < bestTime) {
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
                <<"float"<<vec[bestVecIdx]<<endl;
            cout<<"--------------------------------------------"<<endl;
            gridIdx++;
        }
        blockIdx++;
    }

    delete[] input;

    bestInfo.float_info.mem_write_time = bestTime;
    bestInfo.float_info.mem_write_throughput = bestThroughput;
}

void runBarrier(int experTime) {

    cout<<"----------- Barrier test ------------"<<endl;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
            double tempTime = MAX_TIME;
            for(int i = 0 ; i < experTime; i++) {       
                //--------test map------------
                double percentage;
                testBarrier(                
                input, info,tempTime, percentage, blockSize, gridSize);

                cout<<"blockSize: "<<blockSize<<'\t'
            	<<"gridSize: "<<gridSize<<'\t'
            	<<"barrier time: "<<tempTime<<" ms\t"
            	<<"time per thread: "<<tempTime/(blockSize *  gridSize)<<" ms\t"
            	<<"percentage: "<<percentage <<"%"<<endl;
            }
        }
    }

    delete[] input;
}


double runMap(int experTime, int& bestBlockSize, int& bestGridSize) {
    double bestTime = MAX_TIME;
    bestBlockSize = -1;
    bestGridSize = -1;
    bool res;

    float *input = (float*)malloc(sizeof(float)*dataSize);
    for(int i = 0; i < dataSize;i++) {
        input[i] = 0;
    }

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
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

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
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

    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {
        for(int gridSize = MIN_GRID; gridSize <= MAX_GRID; gridSize <<= 1) {   
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
     
    for(int blockSize = MIN_BLOCK; blockSize <= MAX_BLOCK; blockSize<<=1) {   
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
