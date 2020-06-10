#pragma once

#ifdef __JETBRAINS_IDE__
#include "../CL/cl.h"
#include "opencl_fake.h"
#include "openmp_fake.h"
#endif

/*literal macros*/
#define ERR_HOST_ALLOCATION                 "Failed to allocate the host memory."
#define ERR_WRITE_BUFFER                    "Failed to write to the buffer."
#define ERR_READ_BUFFER                     "Failed to read back the device memory."
#define ERR_SET_ARGUMENTS                   "Failed to set the arguments."
#define ERR_EXEC_KERNEL                     "Failed to execute the kernel."
#define ERR_LOCAL_MEM_OVERFLOW              "Local memory overflow "
#define ERR_COPY_BUFFER                     "Failed to copy the buffer."
#define ERR_RELEASE_MEM                     "Failed to release the device memory object."

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include "utility.h"
#include "../params.h"

/*
 *  define the structure of data
 *  KO: key-only
 *  KVS_AOS: key-value store using Array of Structures (AOS)
 *  KVS_SOA: key-value store using Structure of Arrays (SOA)
 */
enum Data_structure {KO, KVS_AOS, KVS_SOA};

/*
 * WI: work-item level split
 * WG: work-group level split
 * WG_reorder_fixed: work-group level split with fixed-length reorder buffers
 * WG_reorder_varied: work-group level split with varied-length reorder buffers
 *
 * */
enum Algo {WI, WG, WG_fixed_reorder, WG_varied_reorder, Single, Single_reorder};
enum LocalScanType {SERIAL, KOGGE, SKLANSKY, BRENT};
enum MatrixScanType {LM, REG, LM_REG, LM_SERIAL};
enum ReorderType {NO_REORDER, FIXED_REORDER, VARIED_REORDER};

typedef cl_int2 tuple_t;    /*for AOS*/


void checkErr(cl_int status, const char* name, int tag=-1);
void cl_mem_free(cl_mem object);
double clEventTime(const cl_event event);
void add_param(char *param, char *macro, bool has_value=false, int value=-1);

/*create the cl_kernel according to the file name and function name*/
cl_kernel get_kernel(
        cl_device_id device, cl_context context,
        char *file_name, char *func_name, char *params=NULL);

double gather(cl_mem d_source_values, cl_mem d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, int pass);
double scatter(cl_mem d_source_values, cl_mem d_dest_values, int length, cl_mem d_loc, int localSize, int gridSize, int pass);

double scan_fast_out_of_place(cl_mem d_in, cl_mem d_out, int length, int local_size, int grid_size, int R, int L);

double scan_chained(cl_mem d_in, cl_mem d_out, int length, int localSize, int gridSize, int R, int L);
double scan_rss(cl_mem d_in, cl_mem d_out, unsigned length, int local_size, int grid_size);
double scan_rss_single(cl_mem d_in, cl_mem d_out, unsigned length);
double scan_global_single_thread(cl_mem d_in, cl_mem d_out, int length, int tile_size);

/*split algorithms*/
double WI_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets,
        Data_structure structure,
        cl_mem d_in_values=0, cl_mem d_out_values=0,
        int local_size=256, int grid_size=32768);

double WG_split(
        cl_mem d_in, cl_mem d_out, cl_mem d_start,
        int length, int buckets, ReorderType reorder_type,
        Data_structure structure,
        cl_mem d_in_values=0, cl_mem d_out_values=0,
        int local_size=256, int grid_size=32768);

double single_split(
        cl_mem d_in, cl_mem d_out,
        int length, int buckets, bool reorder,
        Data_structure structure);

double partitionHJ(cl_mem d_R, int rLen,int totalCountBits, int localSize, int gridSize) ;
double hashjoin(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len);
double hashjoin_np(cl_mem d_R_keys, cl_mem d_R_values, int rLen, cl_mem d_S_keys, cl_mem d_S_values, int sLen, int &res_len);


//-------------------------test primitives-------------------------
//void test_bandwidth();
//void test_wg_sequence(unsigned long len);
//void testAccess();
//bool testGather(int len);
//bool testScatter(int len, int inputPass);

bool testScanSingleThread(int length, double &aveTime, int tile_size);
bool testScan(int length, double &totalTime, int localSize, int gridSize, int R, int L, bool oop=true);

bool test_scan_local_schemes(LocalScanType type, double &ave_time, unsigned repeat);
bool test_scan_matrix(MatrixScanType type, double &ave_time, int tile_size, unsigned repeat);

/*
 *  Split test function, to test specific kernel configurations
 *
 *  @param len          number of elements of the input dataset
 *  @param info         platform info
 *  @param buckets      number of buckets
 *  @param aveTime      average kernel execution time
 *  @param algo         algorithms being tested
 *  @param structure    data structure of the input dataset
 *  @param local_size   number of WIs in a WG
 *  @param grid_size    number of WGs in the kernel
 *
 * */
void split_test_specific(
        int len, int buckets,
        Algo algo, Data_structure structure,
        int local_size, int grid_size);

/*
 *  Split test function, to probe for the best kernel configuration
 *  of a specific split kernel
 *
 *  @param len          number of elements of the input dataset
 *  @param buckets      number of buckets
 *  @param algo         algorithms being tested
 *  @param structure    data structure of the input dataset
 *  @param device       type of device
 *  @param info         platform info
 *
 * */
void split_test_parameters(
        int len, int buckets,
        Algo algo, Data_structure structure,
        int device);

//-------------------------test joins-------------------------
bool testHj(int rLen, int sLen);


