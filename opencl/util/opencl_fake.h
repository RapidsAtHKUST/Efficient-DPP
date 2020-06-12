/*  Reference: OpenCL V1.2 cheat sheet
 *  https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf
 * */

#pragma once

// 1st: macros
#define kernel
#define global
#define local

#define CLK_LOCAL_MEM_FENCE     0
#define CLK_GLOBAL_MEM_FENCE    1

// 2nd: built-in functions
int get_global_id(int);
int get_global_size(int);
int get_local_id(int);
int get_local_size(int);
int get_group_id(int);
int get_num_groups(int);
void barrier(int);
void mem_fence(int);

/*atomic functions*/
int atomic_inc(int*);
int atomic_add(int*,int);
int atomic_cmpxchg(int* data, int old, int val);

using uint=unsigned int;

