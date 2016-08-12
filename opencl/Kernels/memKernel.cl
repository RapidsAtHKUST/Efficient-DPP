#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//-------------------------------- read operation ---------------------------

kernel void mem_read_float1 (
    global float* restrict d_source_values,
    global float* restrict d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int globalId_static = globalId;

    float v = 0.0;

	while (globalId < length) {
	   v += d_source_values[globalId];
	   globalId += globalSize;
	}
    d_dest_values[globalId_static] = v;
}

kernel void mem_read_float2 (
    global float2* restrict d_source_values,
    global float* restrict d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int globalId_static = globalId;

    float2 v = 0.0;
	int globalId_loop = globalId;
    while (globalId < length/2) {
        v += d_source_values[globalId];
        globalId += globalSize;
    }
    d_dest_values[globalId_static] = v.x + v.y;
}

kernel void mem_read_float4 (
    global float4* restrict d_source_values,
    global float* restrict d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int globalId_static = globalId;

    float4 v = 0.0;
    while (globalId < length/4) {
        v += d_source_values[globalId];
        globalId += globalSize;
    }
    d_dest_values[globalId_static] = v.x + v.y + v.z + v.w;
}

kernel void mem_read_float8 (
    global float8* restrict d_source_values,
    global float* restrict d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int globalId_static = globalId;

    float8 v = 0.0;
    while (globalId < length/8) {
        v += d_source_values[globalId];
        globalId += globalSize;
    }
    d_dest_values[globalId_static] = v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

kernel void mem_read_float16 (
    global float16* restrict d_source_values,
    global float* restrict d_dest_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    int globalId_static = globalId;

    float16 v = 0.0;
    while (globalId < length/16) {
        v += d_source_values[globalId];
        globalId += globalSize;
    }
    d_dest_values[globalId_static] = v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7 + v.s8 + v.s9 + v.sa + v.sb + v.sc + v.sd + v.se + v.sf;
}
//-------------------------------- write operation ---------------------------

kernel void mem_write_float1 (
    global float* d_source_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length) {
    	d_source_values[globalId] = globalId + 0.15;
    	globalId += globalSize;
	}
}

kernel void mem_write_float2 (
    global float2* d_source_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/2) {
    	float2 v = (float2)
    	(globalId+0.15, globalId+0.0131);

    	d_source_values[globalId] = v;
    	globalId += globalSize;
	}
}

kernel void mem_write_float4 (
    global float4* d_source_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/4) {
    	float4 v = (float4)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411);

    	d_source_values[globalId] = v;
    	globalId += globalSize;
	}
}

kernel void mem_write_float8 (
    global float8* d_source_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
   
    while (globalId < length/8) {
    	float8 v = (float8)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411, globalId+99.11, globalId+32.34,globalId+6.45, globalId+976.335);

    	d_source_values[globalId] = v;
    	globalId += globalSize;
	}
    
}

kernel void mem_write_float16 (
    global float16* d_source_values,
    int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    while (globalId < length/16) {
    	float16 v = (float16)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411, globalId+99.11, globalId+32.34,globalId+6.45, globalId+976.335, globalId+908.222, globalId+123.0981, globalId+104.4781, globalId+11.1361,globalId+121.4671, globalId+134.3561,
    		globalId+14.1, globalId+15.1);
    	d_source_values[globalId] = v;
    	globalId += globalSize;
	}
}

// -------------------------------  Triad test -------------------------
// kernel void triad_float1(
//     global float * restrict a,
//     global const float *restrict b,
//     global const float *restrict c,
//     const int length)
// {
//     const int globalId = get_global_id(0);

// //     while (globalId < length) {
// //         a[globalId] = b[globalId] + scalar * c[globalId]; 
// //         globalId += globalSize;
// //     }
//     a[globalId] = b[globalId] + 0.3 * c[globalId];
// }

kernel void triad_float1(
    global float * restrict a,
    global const float * restrict b,
    global const float * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] = b[i] + 0.3 * c[i];
  }



kernel void triad_float2(
    global float2 *restrict a,
    global const float2 *restrict b,
    global const float2 *restrict c,
    const int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    const float2 scalar = 0.3;

    while (globalId < length/2) {
        a[globalId] = b[globalId] + scalar * c[globalId]; 
        globalId += globalSize;
    }
}

kernel void triad_float4(
    global float4 *restrict a,
    global const float4 *restrict b,
    global const float4 *restrict c,
    const int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    const float4 scalar = 0.3;

    while (globalId < length/4) {
        a[globalId] = b[globalId] + scalar * c[globalId]; 
        globalId += globalSize;
    }
}

kernel void triad_float8(
    global float8 *restrict a,
    global const float8 *restrict b,
    global const float8 *restrict c,
    const int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    const float8 scalar = 0.3;

    while (globalId < length/8) {
        a[globalId] = b[globalId] + scalar * c[globalId]; 
        globalId += globalSize;
    }
}

kernel void triad_float16(
    global float16 *restrict a,
    global const float16 *restrict b,
    global const float16 *restrict c,
    const int length)
{
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    const float16 scalar = 0.3;

    while (globalId < length/16) {
        a[globalId] = b[globalId] + scalar * c[globalId]; 
        globalId += globalSize;
    }
}

#endif