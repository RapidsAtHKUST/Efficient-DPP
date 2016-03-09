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

//generate a sorted ascending record array
void recordSorted(Record *records, int length, int max) {
    recordRandom(records, length, max);
    qsort(records, length, sizeof(Record), compRecordAsc);
    for(int i = 0; i < length;i++) {
        records[i].x = i;
    }
}

void recordSorted_Only(Record *records, int length) {
    for(int i = 0; i < length;i++) {
        records[i].x = records[i].y = i;
    }
}

void recordRandom(Record *records, int length, int max) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length ; i++) {
        records[i].x = i;
        records[i].y = rand() % max;
    }
}

void recordRandom_Only(Record *records, int length,  int times) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length ; i++) {
        records[i].x = i;
        records[i].y = i;
    }
    int temp, from = 0, to = 0;
    for(int i = 0; i < times; i++) {
        from = rand() % length;
        to = rand() % length;
        temp = records[from].y;
        records[from].y = records[to].y;
        records[to].y = temp;
    }
}

void intRandom(int *intArr, int length, int max) {
    srand((unsigned)time(NULL));
    for(int i = 0; i < length; i++) {
        intArr[i] = rand() % max;
    }
}

void intRandom_Only(int *intArr, int length,  int times) {
    srand((unsigned)time(NULL));
    sleep(1);
    for(int i = 0; i < length; i++) {
        intArr[i] = i;
    }
    int temp, from = 0, to = 0;
    for(int i = 0; i < times; i++) {
        from = rand() % length;
        to = rand() % length;
        temp = intArr[from];
        intArr[from] = intArr[to];
        intArr[to] = temp;
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


#ifdef OPENCL_PROJ
//OpenCL error checking functions
void checkErr(cl_int status, const char* name) {
    if (status != CL_SUCCESS) {
        std::cout<<"statusError: " << name<< " (" << status <<") "<<std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif





