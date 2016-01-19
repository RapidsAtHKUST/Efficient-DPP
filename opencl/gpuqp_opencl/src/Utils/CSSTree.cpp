//
//  CSSTree.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 5/18/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#include "CSSTree.h"


//search for the leftmost
int bisearch(Record *input, int begin, int end, int obj) {
    int res = -1;
    int mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        if (obj > input[mid].y)   begin = mid+1;
        else {
            if (obj == input[mid].y)  res = mid;
            end = mid-1;
        }
    }
    return res;
}

//search for the leftmost in node
int bisearchInNode(int *input, int begin, int end, int obj) {
    int res = -1;
    int mid;
    
    while (begin <= end) {
        mid = (begin + end)/2;
        if (obj > input[mid])   begin = mid+1;
        else {
            res = mid;
            end = mid-1;
        }
    }
    return res;
}

int calMPart(int length) {
    int sq = (int)sqrt(length);
    int k = 1;
    while (k <= sq)   k<<=1;
    return (k >> 1) - 1;
}

int* generateCSSTree(Record *a, int total, int m, int &numOfInternalNodes, int &mark) {
    
    int B = ceil(1.0 * total / m );
    int k = ceil(log(B)/log(m+1));
    numOfInternalNodes = (pow(m+1, k) - 1)/m - floor((pow(m+1, k) - B)/m) ; 
    int leftMostLeaveNo = (pow(m+1, k) - 1)/m;
    int lastEleInPartOneIdx = total - ( leftMostLeaveNo - numOfInternalNodes ) * m - 1 ;
    mark = leftMostLeaveNo * m ;
    
    int *b = new int[numOfInternalNodes * m];     //index array

    for(int i = numOfInternalNodes * m - 1; i >= 0; i--) {
        int d = i/m;
        int c = d * (m + 1) + 1 + i % m;
        while (c <= (numOfInternalNodes - 1) ) {
            c = c * ( m + 1 ) + m + 1;
        }
        
        int diff = c * m  - mark;
        
        if (diff < 0) {     //map to the second half of the array
            b[i] = a[diff + total + m - 1].y;
        }
        else {              //mao to the first half of the array
            if (diff + m - 1 <= lastEleInPartOneIdx)
                b[i] = a[diff + m - 1].y;
            else
                b[i] = a[lastEleInPartOneIdx].y;
        }
    }
    return  b;
}


int searchInCSSTree(int obj, Record *a, int *b, int num, int m, int numOfInternalNodes, int mark)
{
    int res = -1;
    
    int d = 0;
    while (d <= numOfInternalNodes - 1) {
        int first = d * m;
        int last = d * m + m - 1;
        int nextBr = bisearchInNode(b, first, last, obj);
        if (nextBr != -1)
            d = d * (m + 1) + 1 + nextBr - first;
        else
            d = d * (m + 1) + m + 1;
    }
    
    int diff = d * m  - mark;
    
    if (diff < 0) {
        res = bisearch(a, num + diff, num + diff + m - 1, obj);
    }
    else {
        res = bisearch(a, diff, diff + m - 1, obj);
    }
    
    return  res;
}

