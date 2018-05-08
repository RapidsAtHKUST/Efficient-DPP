/*
 *  compile: icc -o scanTBB -O3 scanTBB.cpp -ltbb
 */
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_scan.h"
#include "tbb/tick_count.h"

using namespace tbb;
using namespace std;

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

bool scanTBB(int length, double &totalTime) {
    bool res = true;
    int *input = new int[length];
    int *output = new int[length];
    for(int i = 0; i < length; i++) input[i] = 1;
    tick_count t1,t2;
    ScanBody_ex<int> body(output,input);

    int experTime = 10;
    int normalCount = 0;
    double normalTempTime;
    for(int e = 0; e < experTime; e++) {
        float tempTime;

        t1=tick_count::now();
        parallel_scan( blocked_range<int>(0,length), body, auto_partitioner());
        t2=tick_count::now();
        tempTime = (t2-t1).seconds()*1000;        //ms

        if(e==0) {      //check
            for(int i = 0; i < length; i++) {
                if (output[i] != i) {
                    res = false;
                    break;
                }
            }
            normalTempTime = tempTime;
        }
        else if (res == true) {
            if (tempTime < normalTempTime*1.05) {      //with 5% error
                if (tempTime*1.05 < normalTempTime) { //means the normalTempTime is an outlier
                    normalCount = 1;
                    normalTempTime = tempTime;
                    totalTime = tempTime;
                }
                else {  //temp time is correct
                    totalTime += tempTime;
                    normalCount++;
                }
            }
        }
        else {
            break;
        }
    }
    totalTime/= normalCount;

    delete[] input;
    delete[] output;

    return res;
};

int main()
{
    for(int scale = 10; scale <= 30; scale++) {
        int length = 1<<scale;
        cout<<"length: "<<length<<'\t';

        double totalTime;
        bool res = scanTBB(length, totalTime);
        if (res) {
            cout<<"Time:"<<totalTime<<" ms"<<'\t'
                <<"Throughput:"<<1.0*length* sizeof(int)/1024/1024/1024/totalTime*1e3<<" GB/s"<<endl;
        }
        else {
            cout<<"wrong results"<<endl;
        }
    }
    return 0;
}