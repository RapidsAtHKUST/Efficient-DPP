//
//  utility.cpp
//  gpuqp_cuda
//
//  Created by Bryan on 01/19/2016.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "utility.h"

int compRecordAsc ( const void * a, const void * b) {
    return ((Record*)a)->y - ((Record*)b)->y;
}

int compRecordDec ( const void * a, const void * b) {
    return ((Record*)b)->y - ((Record*)a)->y;
}

int compInt ( const void * p, const void * q) {
    return  *(int*) p - *(int*)q ;
}

template<typename T>
T getRandNum( T max) {
    //a little bit wierd here because sometimes ( 5 out of 10,0000,0000) it will
    //get a res that is equal to max!
    T res = static_cast<T>(rand())  /  static_cast<T> (RAND_MAX / max);
    if (res >= max) return max-1;
    else            return res;
}

//generate a sorted ascending record array
template<typename T>
void recordSorted(int *keys, T *values, int length, T max) {
    valRandom<T>(values, length, max);
    std::sort(values, values + length);
    for(int i = 0; i < length;i++) {
        keys[i] = i;
    }
}

template<typename T>
void recordSorted_Only(int *keys, T *values, int length) {
    for(int i = 0; i < length;i++) {
        keys[i] = i;
        values[i] = (T)i;
    }
}

template<typename T>
void recordRandom(int *keys, T *values, int length, T max) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length ; i++) {
        keys[i] = i;
        values[i] = getRandNum<T>(max);
    }
}

template<typename T>
void recordRandom_Only(int *keys, T *values, int length,  int times) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length ; i++) {
        keys[i] = i;
        values[i] = (T)i;
    }
    T temp;
    int from = 0, to = 0;
    for(int i = 0; i < times; i++) {
        from = rand() % length;
        to = rand() % length;
        temp = values[from];
        values[from] = values[to];
        values[to] = temp;
    }
}

template<typename T>
void valRandom(T *arr, int length, T max) {
    srand((unsigned)time(NULL));
    for(int i = 0; i < length; i++) {
        arr[i] = getRandNum<T>(max);
        assert(arr[i] >= 0);
    }
}

template<typename T>
void valRandom_Only(T *arr, int length,  int times) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length; i++) {
        arr[i] = i;
    }
    T temp; 
    int from = 0, to = 0;
    for(int i = 0; i < times; i++) {
        from = rand() % length;
        to = rand() % length;
        temp = arr[from];
        arr[from] = arr[to];
        arr[to] = temp;
    }
}

double calCPUTime(clock_t start, clock_t end) {
    return (end - start) / (double)CLOCKS_PER_SEC;
}

void printbinary(const unsigned int val, int dis) {
    int count = 0;
    for(int i = dis; i >= 0; i--)
    {
        if(val & (1 << i))
            std::cout << "1";
        else
            std::cout << "0";
        count ++;
        if (count >= 4) {
            count = 0;
            std::cout<<' ';
        }
    }
}



// void generateFixedRecords(Record* fixedRecords, int length, bool write, char *file) {
//     recordRandom(fixedRecords, length, MAX_NUM);
//     if (!write) return;
    
//     char fileAddr[300] = DATADIR;
//     strcat(fileAddr, file);
//     std::ofstream out(fileAddr,std::ios::binary);
    
//     if(!out.good()) {
//         std::cerr<<"Data directory open failed!"<<std::endl;
//         exit(1);
//     }
    
//     out<<length<<std::endl;
    
//     for(int i = 0; i < length; i++) {
//         out<<fixedRecords[i].x<<' '<<fixedRecords[i].y<<std::endl;
//     }
    
//     out.close();
// }

// void generateFixedArray(int *fixedArray, int length, bool write, char *file) {
//     intRandom(fixedArray, length, INT_MAX);
//     if (!write) return;
    
//     char fileAddr[300] = DATADIR;
//     strcat(fileAddr, file);
//     std::ofstream out(fileAddr,std::ios::binary);
    
//     if(!out.good()) {
//         std::cerr<<"Data directory open failed!"<<std::endl;
//         exit(1);
//     }
    
//     out<<length<<std::endl;
    
//     for(int i = 0; i < length; i++) {
//         out<<fixedArray[i]<<std::endl;
//     }
    
//     out.close();
// }

void readFixedRecords(Record* fixedRecords, char *file, int& recordLength) {
    
    std::ifstream in(file,std::ios::binary);
    
    if (!in.good()) {
        std::cerr<<"Data file not found."<<std::endl;
        exit(1);
    }
    
    in>>recordLength;

    fixedRecords = new Record[recordLength];

    for(int i = 0; i < recordLength; i++) {
        in>>fixedRecords[i].x>>fixedRecords[i].y;
    }
}

void readFixedArray(int* fixedArray, char *file, int & arrayLength) {
    std::ifstream in(file, std::ios::binary);
    
    if (!in.good()) {
        std::cout<<"Data file not found."<<std::endl;
        exit(1);
    }
    
    in>>arrayLength;
    
    fixedArray = new int[arrayLength];
    
    for(int i = 0; i < arrayLength; i++) {
        in>>fixedArray[i];
    }
}

double diffTime(struct timeval end, struct timeval start) {
	return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

int floorOfPower2_CPU(int a) {
	int base = 1;
	while (base < a) {
		base <<= 1;
	}
	return base>>1;
}

void printRes(std::string funcName, bool res, double elaspsedTime) {
    std::cout<<funcName<<": \t\t";
    
    std::cout<<"Time: "<<elaspsedTime<<" ms."<<"\t\t\t";
    if (res)    std::cout<<"Pass!"<<std::endl;
    else        std::cout<<"Failed!"<<std::endl;
}

void printRes(std::string funcName, bool res, float elaspsedTime) {
    std::cout<<funcName<<": \t\t";
    
    std::cout<<"Time: "<<elaspsedTime<<" ms."<<"\t\t\t";
    if (res)    std::cout<<"Pass!"<<std::endl;
    else        std::cout<<"Failed!"<<std::endl;
}

void my_itoa(int num, char *buffer, int base) {
    int len=0, p=num;
    while(p/=base)
    {
        len++;
    }
    len++;
    for(p=0;p<len;p++)
    {
        int x=1;
        for(int t=p+1;t<len;t++)
        {
            x*=base;
        }
        buffer[p] = num/x + '0';
        num -=( buffer[p] - '0' ) * x;
    }
    buffer[len] = '\0';
}


#ifdef OPENCL_PROJ
//OpenCL error checking functions
void checkErr(cl_int status, const char* name) {
    if (status != CL_SUCCESS) {
        std::cout<<"statusError: " << name<< " (" << status <<") "<<std::endl;
        exit(EXIT_FAILURE);
    }
}

double clEventTime(const cl_event event){
    cl_ulong start,end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return (end - start) / 1000000.0;
}



#endif

//templates

//int
template void recordSorted<int>(int *keys, int *values, int length, int max);
template void recordSorted_Only<int>(int *keys, int *values, int length);
template void recordRandom<int>(int *keys, int *values, int length, int max);
template void recordRandom_Only<int>(int *keys, int *values, int length,  int times);
template void valRandom<int>(int *arr, int length, int max);
template void valRandom_Only<int>(int *arr, int length,  int times);

//long
template void recordSorted<long>(int *keys, long *values, int length, long max);
template void recordSorted_Only<long>(int *keys, long *values, int length);
template void recordRandom<long>(int *keys, long *values, int length, long max);
template void recordRandom_Only<long>(int *keys, long *values, int length,  int times);
template void valRandom<long>(long *arr, int length, long max);
template void valRandom_Only<long>(long *arr, int length,  int times);

//float
template void recordSorted<float>(int *keys, float *values, int length, float max);
template void recordSorted_Only<float>(int *keys, float *values, int length);
template void recordRandom<float>(int *keys, float *values, int length, float max);
template void recordRandom_Only<float>(int *keys, float *values, int length,  int times);
template void valRandom<float>(float *arr, int length, float max);
template void valRandom_Only<float>(float *arr, int length,  int times);

//double
template void recordSorted<double>(int *keys, double *values, int length, double max);
template void recordSorted_Only<double>(int *keys, double *values, int length);
template void recordRandom<double>(int *keys, double *values, int length, double max);