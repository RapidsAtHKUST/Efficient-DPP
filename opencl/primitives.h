//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#pragma once

#include "utility.h"
#include "params.h"
#include "types.h"

/*gather algorithm*/
double gather(cl_mem d_source_values, cl_mem d_dest_values,
              int length, cl_mem d_loc, int localSize,
              int gridSize, int pass);

/*scatter algorithm*/
double scatter(cl_mem d_source_values, cl_mem d_dest_values,
               int length, cl_mem d_loc, int localSize,
               int gridSize, int pass);

/*scan algorithms*/
double scan_chained(cl_mem d_in, cl_mem d_out,
                    int length, int localSize,
                    int gridSize, int R, int L);

double scan_RSS(cl_mem d_in, cl_mem d_out,
                unsigned length, int local_size, int grid_size);

double scan_RSS_single(cl_mem d_in, cl_mem d_out, unsigned length);

/*split algorithms*/
double WI_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets,
        DataStruc structure,
        cl_mem d_in_values=0, cl_mem d_out_values=0,
        int local_size=256, int grid_size=32768);

double WG_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets, ReorderType reorder_type,
        DataStruc structure,
        cl_mem d_in_values=0, cl_mem d_out_values=0,
        int local_size=256, int grid_size=32768);

double single_split(
        cl_mem d_in, cl_mem d_out,
        int length, int buckets, bool reorder,
        DataStruc structure);


