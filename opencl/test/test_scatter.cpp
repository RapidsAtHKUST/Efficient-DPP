//
//  main.cpp
//  gpuqp_opencl
//
//  Created by Bryan on 4/7/15.
//  Copyright (c) 2015 Bryan. All rights reserved.
//
#include "Plat.h"

using namespace std;

int main(int argc, const char *argv[]) {

    Plat::plat_init();

    double totalTime;
    int repeat = 1;
    bool res;

    int inputPass = atoi(argv[1]);

    for(int i = 2048; i <= 2048; i += 256) {
        int num = i / sizeof(int) * 1024 * 1024;
        testScatter(num, inputPass);
        cout<<endl;
    }

    return 0;
}
