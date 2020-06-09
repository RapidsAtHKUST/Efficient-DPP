#include "../util/Plat.h"
using namespace std;

int main(int argc, char *argv[])
{
    Plat::plat_init();

    double totalTime;
    uint32_t repeat = 1000;
    bool res;

    /*local matrix scan*/
    for(int tile_size = 1; tile_size < 50; tile_size++)
    {
        cout<<"-------- Tile = "<<tile_size<<" --------"<<endl;
        cout << "Matrix_LM: ";
        res = test_scan_matrix(LM, totalTime, tile_size, repeat);
        if (res) cout << "right" << ' ';
        else cout << "wrong" << ' ';
        cout << "total time:" << totalTime << "ms\t"
             << " (repeat " << repeat << " times)\t"
             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
             << endl;

        cout << "Matrix_REG: ";
        res = test_scan_matrix(REG, totalTime, tile_size, repeat);
        if (res) cout << "right" << ' ';
        else cout << "wrong" << ' ';
        cout << "total time:" << totalTime << "ms\t"
             << " (repeat " << repeat << " times)\t"
             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
             << endl;

        cout << "Matrix_LM_REG: ";
        res = test_scan_matrix(LM_REG, totalTime, tile_size, repeat);
        if (res) cout << "right" << ' ';
        else cout << "wrong" << ' ';
        cout << "total time:" << totalTime << "ms\t"
             << " (repeat " << repeat << " times)\t"
             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
             << endl;

        cout << "Matrix_LM_SERIAL: ";
        res = test_scan_matrix(LM_SERIAL, totalTime, tile_size, repeat);
        if (res) cout << "right" << ' ';
        else cout << "wrong" << ' ';
        cout << "total time:" << totalTime << "ms\t"
             << " (repeat " << repeat << " times)\t"
             << "Throughput: "<< 512*tile_size/totalTime*1000*1000/1024/1024<<" MKeys/sec"
             << endl;
    }

    return 0;
}