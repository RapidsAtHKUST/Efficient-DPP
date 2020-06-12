//
//  Created by Zhuohang Lai on 4/7/15.
//  Copyright (c) 2015 Zhuohang Lai. All rights reserved.
//
#pragma once

/*scan types*/
enum SCAN_GLOBAL_TYPE {
    RSS, RSS_SINGLE_THREAD, CHAINED
};

enum MatrixScanType {
    LM, REG, LM_REG, LM_SERIAL
};

struct scan_arg {
    int R;  //number of values per work-item stored in registers
    int L;  //number of values per work-item stored in local memory
    SCAN_GLOBAL_TYPE algo;
};

/*split types*/
/*
 *  define the structure of data
 *  KO: key-only
 *  KVS_AOS: key-value store using Array of Structures (AOS)
 *  KVS_SOA: key-value store using Structure of Arrays (SOA)
 */
enum DataStruc {
    KO, KVS_AOS, KVS_SOA
};

/*
 * WI: work-item level split
 * WG: work-group level split
 * WG_reorder_fixed: work-group level split with fixed-length reorder buffers
 * WG_reorder_varied: work-group level split with varied-length reorder buffers
 *
 * */
enum SPLIT_ALGO {
    WI, WG, WG_fixed_reorder, WG_varied_reorder, Single, Single_reorder
};

enum ReorderType {
    NO_REORDER, FIXED_REORDER, VARIED_REORDER
};

typedef cl_int2 tuple_t;    /*for AOS*/