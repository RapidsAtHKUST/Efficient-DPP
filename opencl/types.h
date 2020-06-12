//
// Created by Bryan on 12/6/2020.
//

#pragma once

/*scan types*/
enum SCAN_GLOBAL_TYPE {
    RSS, RSS_SINGLE_THREAD, CHAINED
};

struct scan_arg {
    int R;  //number of values per work-item stored in registers
    int L;  //number of values per work-item stored in local memory
    SCAN_GLOBAL_TYPE algo;
};
