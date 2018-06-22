#include <iostream>
#include <omp.h>
#include <sys/time.h>
using namespace std;

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

int main() {



    return 0;
}