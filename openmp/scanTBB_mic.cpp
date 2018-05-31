/*
 * build:
 * 1. set the host env variable by using:
 *      source /usr/local/intel/bin/compilervars.sh intel64 (setting the MIC_LD_LIBRARY and MIC_LIBRARY envs)
 * 2. compile the file using:
 *      icc -O3 -o scanTBB_mic scanTBB_mic.cpp -tbb
 */
//all the header files should be put into the offload area!!
#pragma offload_attribute(push, target(mic))
#include <iostream>
#include <sys/time.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/parallel_for.h"
using namespace tbb;

//inclusive scan reaches 80% utilization
template<typename T>
class ScanBody_in {
    T sum;
    T* const y;
    const T* const x;
public:
    ScanBody_in( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
    T get_sum() const {return sum;}

    template<typename Tag>
    void operator()( const blocked_range<int>& r, Tag ) {
        T temp = sum;
        int end = r.end();
        for( int i=r.begin(); i<end; ++i ) {
            temp = temp + x[i];
            if( Tag::is_final_scan() )
                y[i] = temp;
        }
        sum = temp;
    }
    ScanBody_in( ScanBody_in& b, split ) : x(b.x), y(b.y), sum(0) {}
    void reverse_join( ScanBody_in& a ) { sum = a.sum + sum;}
    void assign( ScanBody_in& b ) {sum = b.sum;}
};

//exclusive scan
template<typename T>
class ScanBody_ex {
    T sum;
    T* const y;
    const T* const x;
public:
    ScanBody_ex( T y_[], const T x_[] ) : sum(0), x(x_), y(y_) {}
    T get_sum() const {return sum;}

    template<typename Tag>
    void operator()( const blocked_range<int>& r, Tag ) {
        T temp = sum;
        int end = r.end();
        for( int i=r.begin(); i<end; ++i ) {
            if( Tag::is_final_scan() )
                y[i] = temp;
            temp = temp + x[i];
        }
        sum = temp;
    }
    ScanBody_ex( ScanBody_ex& b, split ) : x(b.x), y(b.y), sum(0) {}
    void reverse_join( ScanBody_ex& a ) { sum = a.sum + sum;}
    void assign( ScanBody_ex& b ) {sum = b.sum;}
};
#pragma offload_attribute(pop)

double diffTime(struct timeval end, struct timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

bool scanTBB_mic(int len, double &totalTime) {
    totalTime = 0;
    bool res = true;

    int *input = (int*)_mm_malloc(sizeof(int)*len, 64);
    int *output = (int*)_mm_malloc(sizeof(int)*len, 64);
    for(int i = 0; i < len; i++) input[i] = 1;

    int experTime = 100;
    int normalCount = 0;
    double normalTempTime;
    for(int e = 0; e < experTime; e++) {
        struct timeval start, end;

        #pragma offload target(mic) \
        in(input:length(len) alloc_if(1) free_if(0)) \
        out(output:length(len) alloc_if(1) free_if(0))
        {
            ScanBody_ex<int> body(output,input);
            gettimeofday(&start, NULL);
            parallel_scan(blocked_range<int>(0,len), body, auto_partitioner());
            gettimeofday(&end, NULL);
        }
        double tempTime = diffTime(end, start);

        //free the device memory
        #pragma offload target(mic) \
        nocopy(input:length(len) free_if(1)) \
        nocopy(output:length(len) free_if(1))
        {}

        if(e==0) {      //check
            for(int i = 0; i < len; i++) {
                if (output[i] != i) {
                    res = false;
                    break;
                }
            }
            // normalTempTime = tempTime;
            // totalTime = tempTime;
        }
        else if (res == true) {
            // if (tempTime < normalTempTime*1.05) {      //with 5% error
            //     if (tempTime*1.05 < normalTempTime) { //means the normalTempTime is an outlier
            //         normalCount = 1;
            //         normalTempTime = tempTime;
            //         totalTime = tempTime;
            //     }
            //     else {  //temp time is correct
            //         totalTime += tempTime;
            //         normalCount++;
            //     }
            // }
            if (e >= experTime/2) {
                totalTime += tempTime;
                normalCount++;
            }
        }
        else {
            break;
        }
    }
    totalTime/= normalCount;

    _mm_free(input);
    _mm_free(output);
    return res;
};

int main()
{
    for(int scale = 10; scale <= 30; scale++) {
        int length = 1<<scale;

        std::cout<<scale<<" length: "<<length<<'\t';

        double totalTime;
        bool res = scanTBB_mic(length, totalTime);
        if (res) {
            std::cout<<"Time:"<<totalTime<<" ms"<<'\t'
                <<"Throughput:"<<1.0*length* sizeof(int)/1024/1024/1024/totalTime*1e3<<" GB/s"<<std::endl;
        }
        else {
            std::cout<<"wrong results"<<std::endl;
        }
    }
    return 0;
}
