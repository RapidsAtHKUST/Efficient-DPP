# OpenCL-Primitives

## Description

OpenCL-based data-parallel primitives (including gather, scatter, scan and split) on heterogenous processors (GPUs, CPUs and 1st generation MICs).

## OpenCL-Primitives

This project uses CMake for compilation.

### Compilation

```
cd opencl
mkdir -p build && cd build
cmake -DOpenCL_LIBRARY="PATH_TO_OPENCL_LIBRARY" .. (the address of libOpenCL.so)
make -j
```

If CUDA is installed, libOpenCL.so is usually located at cuda/targets/x86_64-linux/lib. The OpenCL drivers for Intel CPUs and MICs should be installed manually if running the code on CPUs and MICs.

### Tests

```./test_access``` : test the performance of column-major order, row-major order and mixed order sequential access patters

```./test_access_wg DATA_NUM``` : test the work-group-wise sequential access efficiency on specified number of input data

```./test_bandwidth ``` : test the sequential bandwidth with the Stream Benchmark (copy, scalar, addition and triad operations)

```./test_gather DATA_NUM``` : test the performance of multi-pass gather on specified number of data

```./test_scatter DATA_NUM``` : test the performance of multi-pass scatter on specified number of data

```./test_scan_local ``` : test the performance of local scan schemes

```./test_scan_global ``` : test the performance of global scan schemes

```./test_split ``` : test the performance of split

## CUDA-Primitives

### Compilation

This project uses CMake for compilation.

```
cd cuda
mkdir -p build && cd build
cmake ..
make -j
```

### Tests

```./test_bandwidth``` : test the sequential bandwidth with the Stream Benchmark (copy, scalar, addition and triad operations)

```./test_gather DATA_NUM``` : test the performance of multi-pass gather on specified number of data

```./test_scatter DATA_NUM``` : test the performance of multi-pass scatter on specified number of data

```./test_scan ``` : test the performance of CUB scan

```./test_split ``` : test the performance of GPU multi-split

## OpenMP-Primitives

### Compilation 

This project uses CMake for compilation.

```
cd cuda
mkdir -p build && cd build
cmake ..
make -j
```

### Tests

```./test_bandwidth_CPU``` : test the sequential bandwidth with the Stream Benchmark (copy and scalar)

```./test_gather_scatter_CPU DATA_NUM``` : test the performance of gather and scatter on specified number of data

```./test_scan_CPU ``` : test the performance of OpenMP-based SSA, RTS scan and TBB scan






