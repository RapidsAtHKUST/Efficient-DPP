#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cub/device/device_scan.cuh"
using namespace std;
using namespace cub;

bool testRadixSort_cub(int len, int buckets) {
    int *h_key_in = new int[len];
    srand(time(NULL));
    for(int i = 0; i < len; i++)    h_key_in[i] = rand() % buckets;

    int *d_key_in;
    cudaMalloc(&d_key_in, len*sizeof(int));
    cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent start_sort, stop_sort;
    cudaEventCreate(&start_sort);
    cudaEventCreate(&stop_sort);

    void *d_temp_storage_sort = NULL;
    size_t 	temp_storage_bytes_sort = 0;

    CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements, 0, buckets));
    cudaMalloc(&d_temp_storage_sort, temp_storage_bytes_sort);

    cudaEventRecord(start_sort, 0);
    cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements));
    cudaEventRecord(stop_sort, 0);
    cudaEventSynchronize(stop_sort);
    cudaEventElapsedTime(&temp_time, start_sort, stop_sort);
    sort_time += temp_time;

    if (h_key_in)   delete[] h_key_in;
}

bool testRadixSort_kv_cub(int len) {

}

if(mode == 0){
random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();

// key-only sort:
void *d_temp_storage_sort = NULL;
size_t 	temp_storage_bytes_sort = 0;

CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements, 0, int(ceil(log2(float(kNumBuckets))))));
CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_sort, temp_storage_bytes_sort));

cudaEventRecord(start_sort, 0);
cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements, 0, int(ceil(log2(float(kNumBuckets)))));
cudaEventRecord(stop_sort, 0);
cudaEventSynchronize(stop_sort);
cudaEventElapsedTime(&temp_time, start_sort, stop_sort);
sort_time += temp_time;

printf("CUB's radix sort finished in %.3f ms, %.3f Mkey/s\n", sort_time, float(n_elements)/sort_time/1000.0f);

if(validate)
{
h_cpu_results_key = new uint32_t[n_elements];
cpu_multisplit_general(h_key_in, h_cpu_results_key, n_elements, bucket_identifier, 0, kNumBuckets);
cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
bool correct = true;
for(int i = 0; i<n_elements && correct;i++)
{
if(h_cpu_results_key[i] != h_gpu_results_key[i]){
printf("### Wrong results at index %d: cpu = %d, gpu = %d\n", i, h_cpu_results_key[i], h_gpu_results_key[i]);
correct = false;
}
}
printf("Validation was done successfully!\n");
}
if(d_temp_storage_sort)CubDebugExit(g_allocator.DeviceFree(d_temp_storage_sort));
}
else if(mode == 10)
{
random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
for(int k = 0; k<n_elements;k++)
h_value_in[k] = h_key_in[k];
cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
cudaMemcpy(d_value_in, h_value_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();

// key-value sort:
void 		*d_temp_storage_sort_pairs = NULL;
size_t 	temp_storage_bytes_sort_pairs = 0;

CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs, d_key_in, d_key_out, d_value_in, d_value_out, n_elements, 0, int(ceil(log2(float(kNumBuckets))))));
CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs));

cudaEventRecord(start_sort, 0);
cub::DeviceRadixSort::SortPairs(d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs, d_key_in, d_key_out, d_value_in, d_value_out, n_elements, 0, int(ceil(log2(float(kNumBuckets)))));
cudaEventRecord(stop_sort, 0);
cudaEventSynchronize(stop_sort);
cudaEventElapsedTime(&temp_time, start_sort, stop_sort);
sort_time += temp_time;

printf("CUB's key-value radix sort finished in %.3f ms, %.3f Mkey/s\n", sort_time, float(n_elements)/sort_time/1000.0f);

if(validate)
{
h_cpu_results_key = new uint32_t[n_elements];
h_cpu_results_value = new uint32_t[n_elements];
cpu_multisplit_pairs_general(h_key_in, h_cpu_results_key, h_value_in, h_cpu_results_value, n_elements, bucket_identifier, 0, kNumBuckets);
cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
cudaMemcpy(h_gpu_results_value, d_value_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
bool correct = true;
for(int i = 0; i<n_elements && correct;i++)
{
if((h_cpu_results_key[i] != h_gpu_results_key[i]) || (h_cpu_results_value[i] != h_gpu_results_value[i])){
printf("### Wrong results at index %d: cpu = (%d,%d), gpu = (%d,%d)\n", i, h_cpu_results_key[i], h_cpu_results_value[i], h_gpu_results_key[i], h_gpu_results_value[i]);
correct = false;
}
}
printf("Validation was done successfully!\n");
}
if(d_temp_storage_sort_pairs)CubDebugExit(g_allocator.DeviceFree(d_temp_storage_sort_pairs));
}