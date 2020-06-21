/*
 * Execute on CPU:
 * 1. Compile the file using:
 *      icc -O3 -o scan_tbb_cpu scan_tbb.cpp -ltbb
 * 2. Execute:
 *      ./scan_tbb_cpu
 *
 * Execute on MIC (only native execution mode):
 * 1. source the system environment
 *      source /usr/local/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
 * 2. Complile the file:
 *      icc -mmic -O3 -o scan_tbb_mic scan_tbb.cpp -ltbb
 * 3. Copy the executable file to MIC:
 *      scp scan_tbb_mic mic0:~
 * 4. (optional) If the MIC does not have libtbb.so.2, copy the library from .../intel/tbb/lib/mic to MIC:
 *      e.g.: scp libtbb.so.2 mic0:~
 * 5. (optional) Set the library path on MIC:
 *      e.g.: export LD_LIBRARY_PATH=~
 * 6. Execute:
 *      ./scan_tbb_mic
 */
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <vector>
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

double scan_tbb(int *input, int* output, int len) {
    bool res = true;
    struct timeval start, end;
    ScanBody_ex<int> body(output,input);

    gettimeofday(&start, NULL);
    parallel_scan(blocked_range<int>(0,len), body, auto_partitioner());
    gettimeofday(&end, NULL);

    return diffTime(end, start);
};

void test_scan_tbb() {
    for(int scale = 10; scale <= 30; scale++) {
        bool res = true;
        int length = 1<<scale;
        std::cout<<scale<<" length: "<<length<<'\t';

        int experTime = 10;
        double tempTimes[experTime];
        for(int e = 0; e < experTime; e++) {
            int *input = new int[length];
            int *output = new int[length];
            dataInitialization(input,length,10);

            //exclusive scan
            tempTimes[e] = scan_tbb(input, output, length);

            if (e == 0) {         //check
                int acc = 0;
                for (int i = 0; i < length; i++) {
                    if (output[i] != acc) {
                        res = false;
                        break;
                    }
                    acc += input[i];
                }
            }

            if(input)   delete[] input;
            if(output)  delete[] output;
        }
        double aveTime = average_Hampel(tempTimes, experTime);

        if (res)
            std::cout<<"Time:"<<aveTime<<" ms"<<'\t'
                     <<"Throughput:"<<1.0*length/1024/1024/1024/aveTime*1e3<<" Gkeys/s"<<std::endl;
        else std::cout<<"wrong results"<<std::endl;
    }
}

int main()
{
    test_scan_tbb();
    return 0;
}