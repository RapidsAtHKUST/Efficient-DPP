//
//  dataDefinition.h
//  gpuqp_opencl
//
//  Created by Bryan on 4/10/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//

#ifndef gpuqp_opencl_dataDefinition_h
#define gpuqp_opencl_dataDefinition_h

#ifdef KERNEL
    typedef int2 Record;
#else
    typedef cl_int2 Record;
#endif

#endif
