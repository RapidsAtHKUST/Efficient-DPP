//
//  generation.cpp
//  comparison_gpu
//
//  Created by Bryan on 01/20/16.
//  Copyright (c) 2015-2016 Bryan. All rights reserved.
//
#include "generation.h"

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

void recordSorted_Only(Record *records, int length, int times) {
	recordRandom_Only(records,length,times);
	qsort(records, length, sizeof(Record), compRecordAsc);
    for(int i = 0; i < length;i++) {
        records[i].x = i;
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

void writeRecords(Record* records, int length, char *file) {
    std::ofstream out(file,std::ios::binary);
    
    if(!out.good()) {
        std::cerr<<"Data directory open failed!"<<std::endl;
        exit(1);
    }
    
    out<<length<<std::endl;
    
    for(int i = 0; i < length; i++) {
        out<<records[i].x<<' '<<records[i].y<<std::endl;
    }
    
    out.close();
}

void writeArray(int *intArr, int length, char *file) {
    std::ofstream out(file,std::ios::binary);
    
    if(!out.good()) {
        std::cerr<<"Data directory open failed!"<<std::endl;
        exit(1);
    }
    
    out<<length<<std::endl;
    
    for(int i = 0; i < length; i++) {
        out<<intArr[i]<<std::endl;
    }
    
    out.close();
}

bool processBool(const char *arg) {
    if (!strcmp(arg,"true")) {
        return true;
    }
    else if (!strcmp(arg,"false")) {
        return false;
    }
    else {
        std::cerr<<"wrong bool parameter."<<std::endl;
        exit(1);
    }
}
