//
//  inljImpl.h
//  gpuqp_opencl
//
//  Created by Bryan on 5/14/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef __gpuqp_opencl__inljImpl__
#define __gpuqp_opencl__inljImpl__

#include "Foundation.h"

//information of CSS tree
typedef struct CSS_Tree_Info {
    
    cl_mem d_CSS;
    int CSS_length;
    int mPart;
    int numOfInternalNodes;
    int mark;
    
} CSS_Tree_Info;

double inlj(cl_mem d_R, int rLen, cl_mem d_S, int sLen, cl_mem& d_Out, int & oLen, CSS_Tree_Info treeInfo, PlatInfo info, int localSize, int gridSize);

#endif /* defined(__gpuqp_opencl__inljImpl__) */
