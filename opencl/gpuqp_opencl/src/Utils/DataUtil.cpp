//
//  DataUtil.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "DataUtil.h"

int compRecordAsc ( const void * a, const void * b) {
    return ((Record*)a)->y - ((Record*)b)->y;
}

int compRecordDec ( const void * a, const void * b) {
    return ((Record*)b)->y - ((Record*)a)->y;
}

int compInt ( const void * p, const void * q) {
    return  *(int*) p - *(int*)q ;
}

double diffTime(struct timeval end, struct timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
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

void intSorted(int *intArr, int length, int max) {
    intRandom(intArr,length,max);
    qsort(intArr,length,sizeof(int),compInt);
}

void intSorted_Only(int *intArr, int length, int times) {
    intRandom_Only(intArr,length,times);
    qsort(intArr,length,sizeof(int),compInt);
}

void checkErr(cl_int status, const char* name) {
    if (status != CL_SUCCESS) {
        std::cout<<"statusError: " << name<< " (" << status <<") "<<std::endl;
        exit(EXIT_FAILURE);
    }
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

void readFixedRecords(Record* fixedRecords, char *file, int& recordLength) {
    
    std::ifstream in(file,std::ios::binary);
    
    if (!in.good()) {
        std::cerr<<"Data file not found."<<std::endl;
        exit(1);
    }
    
    in>>recordLength;

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
    
    for(int i = 0; i < arrayLength; i++) {
        in>>fixedArray[i];
    }
}

int floorOfPower2(int a) {
    int base = 1;
    while (base < a) {
        base <<=1;
    }
    return base >>1;
}

