#ifndef __PARAMS_H__
#define __PARAMS_H__

#ifdef __JETBRAINS_IDE__
#include "CL/cl.h"
#include "util/opencl_fake.h"
#endif

#define WARP_BITS                   (4)
#define WARP_SIZE                   (1<<(WARP_BITS))
#define MASK                        (WARP_SIZE-1)

#define SPLIT_VALUE_DEFAULT         (1024)       /*default value*/
#define EXPERIMENT_TIMES            (10)

#endif