//
//  CSSTree.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/18/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef gpuqp_opencl_CSSTree_h
#define gpuqp_opencl_CSSTree_h

#include "Foundation.h"

int* generateCSSTree(Record *a, int total, int m, int &numOfInternalNodes, int &mark) ;
int searchInCSSTree(int obj, Record *a, int *b, int total, int m, int numOfInternalNodes, int mark);
int calMPart(int length) ;

#endif
