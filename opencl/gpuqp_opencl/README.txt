/*
Source code for gpuqp on OpenCL
May, 2015
*/

***GENEERAL INFORMATION***
There are Five directories and one Makefile in the main project file.  

***ENVIRONMENT***
- Hardware: GPUs or coprocessors that support OpenCL 1.1 API.
- Operating System: Linux 12.04 LTS or compatible OS
- Software: SDKs for running devices supporting OpenCL 1.1
- Programming IDE: Xcode 6.3.2 or compatible compilers.

***PACKAGE DESCRIPTION***
- Header Files:
DataUtil.h: All the macro definitions and data processing functions are defined in this header file. 
KernelProcessor.h: Define the OpenCL platform helping functions, especially for kernel reading and compilation.
PlatInit.h: Descirbe the platform currently used. Get the running context, command queues from the chosen platform.
gpuHeaders.h: List all the primitive and join header files.
Foundation.h: It includes the OpenCL platform encapsulated APIs defined in KernelProcessor.h and PlatInit.h.
Others: Each cpp files has a corresponding header file sharing its implemented functions.

- Primitives:
There are six kinds of primitives implemented: map, gather, scatter, split, scan and sort. For sorting, both bitonic sort and radix sort are provided.

- Join Algorithm:
 - Four kinds of joins algorithms are included: Non-Indexed Nested Loop Joins, Indexed Nested Loop Joins, Sort-Merge Joins and Hash Joins.

***TEST CODE***
Test codes are included in /Test/testJoins.cpp and /Test/testPrimitives.cpp. By default, the executable file lauches both of them.

***TO RUN***
- Simply run the Makefile. Add "PRINT=1" after "make" if you want to see the OpenCL kernel execution order. Then execute the gpuqp in /bin. Configs of data set numbers can be accessed in /src/Main/main.cpp. Configs of work item number and work group number can be accessed in /inc/DataUtil.h.
