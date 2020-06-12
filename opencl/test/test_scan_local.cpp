#include "../util/Plat.h"
#include "log.h"
using namespace std;

/*fixed on 512 * TILE_SIZE elements*/
bool test_scan_matrix(MatrixScanType type, double &ave_time, int tile_size, unsigned repeat) {
    device_param_t param = Plat::get_device_param();

    cl_event event;
    cl_int status = 0;
    bool res = true;

    int local_size = 512;   /*fixed to 512 WIs*/
    int grid_size = 1;      /*single WG*/
    int len_total = local_size*tile_size;

    int *h_input = new int[len_total];
    int *h_output = new int[len_total];
    cl_mem d_in, d_out;

    srand(time(nullptr));
    for(int i = 0; i < len_total; i++) h_input[i] = rand() & 0xf;

    double tempTimes[EXPERIMENT_TIMES];
    for(int e = 0; e < EXPERIMENT_TIMES; e++) {
        d_in = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * len_total, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);
        d_out = clCreateBuffer(param.context, CL_MEM_READ_WRITE, sizeof(int) * len_total, nullptr, &status);
        checkErr(status, ERR_HOST_ALLOCATION);

        status = clEnqueueWriteBuffer(param.queue, d_in, CL_TRUE, 0, sizeof(int) * len_total, h_input, 0, 0, 0);
        checkErr(status, ERR_WRITE_BUFFER);
        clFinish(param.queue);

        char paras[500];
        sprintf(paras, "-DTILE_SIZE=%d", tile_size);

        /*kernel setting*/
        char kernel_name[100] = {'\0'};

        switch (type) {
            case LM:
                strcpy(kernel_name, "matrix_scan_lm");
                break;
            case REG:
                strcpy(kernel_name, "matrix_scan_reg");
                break;
            case LM_REG:
                strcpy(kernel_name, "matrix_scan_lm_reg");
                break;
            case LM_SERIAL:
                strcpy(kernel_name, "matrix_scan_lm_serial");
                break;
        }
        cl_kernel local_scan_kernel = get_kernel(param.device, param.context, "scan_local_kernel.cl", kernel_name, paras);

        size_t local_dim[1] = {(size_t)local_size};
        size_t global_dim[1] = {(size_t)(local_size*grid_size)};

        int argsNum = 0;
        status |= clSetKernelArg(local_scan_kernel, argsNum++, sizeof(cl_mem), &d_in);
        status |= clSetKernelArg(local_scan_kernel, argsNum++, sizeof(cl_mem), &d_out);

        if (type != REG)
            status |= clSetKernelArg(local_scan_kernel, argsNum++, sizeof(int) * local_size * tile_size, nullptr);
        if (type != LM_SERIAL)
            status |= clSetKernelArg(local_scan_kernel, argsNum++, sizeof(int) * local_size, nullptr);

        status = clFinish(param.queue);
        struct timeval start, end;
        gettimeofday(&start, nullptr);
        for(int q = 0; q < repeat; q++) {
            status = clEnqueueNDRangeKernel(param.queue, local_scan_kernel, 1, 0, global_dim, local_dim, 0, nullptr, &event);
            checkErr(status, ERR_EXEC_KERNEL);
            status = clFinish(param.queue);
        }
        gettimeofday(&end, nullptr);
        double temp_time = diffTime(end, start);

        status = clEnqueueReadBuffer(param.queue, d_out, CL_TRUE, 0, sizeof(int)*len_total, h_output, 0, nullptr, nullptr);
        checkErr(status, ERR_READ_BUFFER);
        status = clFinish(param.queue);

        /*check*/
        if (e == 0) {
            int acc = 0;
            for (int i = 0; i < len_total; i++) {
                if (h_output[i] != acc) {
                    res = false;
                    break;
                }
                acc += h_input[i];
            }
        }
        tempTimes[e] = temp_time;

        cl_mem_free(d_in);
        cl_mem_free(d_out);
    }
    ave_time = average_Hampel(tempTimes, EXPERIMENT_TIMES);

    if(h_input) delete[] h_input;
    if(h_output) delete[] h_output;

    return res;
}

int main(int argc, char *argv[]) {
    Plat::plat_init();

    double totalTime;
    uint32_t repeat = 1000;
    bool res;

    /*local matrix scan*/
    for(int tile_size = 1; tile_size < 50; tile_size++) {
        log_info("-------- Tiles = %d  --------", tile_size);
        res = test_scan_matrix(LM, totalTime, tile_size, repeat);
        if (!res) log_error("Wrong results");
        log_info("Func: Matrix_LM, total time = %.1f ms, (repeat %d times), throughput = %.1f MKyes/sec",
        totalTime, repeat, 512.0*tile_size/totalTime*1000*1000/1024/1024);

        res = test_scan_matrix(REG, totalTime, tile_size, repeat);
        if (!res) log_error("Wrong results");
        log_info("Func: Matrix_REG, total time = %.1f ms, (repeat %d times), throughput = %.1f MKyes/sec",
                 totalTime, repeat, 512.0*tile_size/totalTime*1000*1000/1024/1024);

        res = test_scan_matrix(LM_REG, totalTime, tile_size, repeat);
        if (!res) log_error("Wrong results");
        log_info("Func: Matrix_LM_REG, total time = %.1f ms, (repeat %d times), throughput = %.1f MKyes/sec",
                 totalTime, repeat, 512.0*tile_size/totalTime*1000*1000/1024/1024);

        res = test_scan_matrix(LM_SERIAL, totalTime, tile_size, repeat);
        if (!res) log_error("Wrong results");
        log_info("Func: Matrix_LM_SERIAL, total time = %.1f ms, (repeat %d times), throughput = %.1f MKyes/sec",
                 totalTime, repeat, 512.0*tile_size/totalTime*1000*1000/1024/1024);
    }

    return 0;
}