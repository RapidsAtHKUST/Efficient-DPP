#ifndef MEM_KERNEL_CL
#define	MEM_KERNEL_CL

//-------------------------------- read operation ---------------------------

kernel void mem_read_write1 (
    global TYPE* restrict d_source_values,
    global TYPE* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId];
}

kernel void mem_read_write2 (
    global TYPE2* restrict d_source_values,
    global TYPE2* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId];
}

kernel void mem_read_writ4 (
    global TYPE4* restrict d_source_values,
    global TYPE4* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId];
}

kernel void mem_read_write8 (
    global TYPE8* restrict d_source_values,
    global TYPE8* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId];
}

kernel void mem_read_write16 (
    global TYPE16* restrict d_source_values,
    global TYPE16* restrict d_dest_values)
{
    int globalId = get_global_id(0);
    d_dest_values[globalId] = d_source_values[globalId];
}
//-------------------------------- write operation ---------------------------

kernel void mem_write1 (global TYPE* d_source_values)
{
    int globalId = get_global_id(0);
    d_source_values[globalId] = globalId + 0.15;
}

kernel void mem_write2 (global TYPE2* d_source_values)
{
    int globalId = get_global_id(0);
    d_source_values[globalId] = (TYPE2)
        (globalId+0.15, globalId+0.0131);
}

kernel void mem_write4 (global TYPE4* d_source_values)
{
    int globalId = get_global_id(0);
    d_source_values[globalId] = (TYPE4)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411);
}

kernel void mem_write8 (global TYPE8* d_source_values)
{
   int globalId = get_global_id(0);
    d_source_values[globalId] = (TYPE8)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411, globalId+99.11, globalId+32.34,globalId+6.45, globalId+976.335);
}

kernel void mem_write16 (global TYPE16* d_source_values)
{
    int globalId = get_global_id(0);
    d_source_values[globalId] = (TYPE16)
    	(globalId+0.15, globalId+0.0131, globalId+22.13, globalId+33.411, globalId+99.11, globalId+32.34,globalId+6.45, globalId+976.335, globalId+908.222, globalId+123.0981, globalId+104.4781, globalId+11.1361,globalId+121.4671, globalId+134.3561,
    		globalId+14.1, globalId+15.1);
}

// -------------------------------  Triad test -------------------------

kernel void triad1(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
{
    int globalId = get_global_id(0);
    a[globalId] = b[globalId] + 0.3 * c[globalId];
}

kernel void triad2(
    global TYPE2 *restrict a,
    global const TYPE2 *restrict b,
    global const TYPE2 *restrict c)
{
    int globalId = get_global_id(0);
    a[globalId] = b[globalId] + 0.3f * c[globalId]; 
}

kernel void triad4(
    global TYPE4 *restrict a,
    global const TYPE4 *restrict b,
    global const TYPE4 *restrict c)
{
    int globalId = get_global_id(0);
    a[globalId] = b[globalId] + 0.3f * c[globalId]; 
}

kernel void triad8(
    global TYPE8 *restrict a,
    global const TYPE8 *restrict b,
    global const TYPE8 *restrict c)
{
    int globalId = get_global_id(0);
    a[globalId] = b[globalId] + 0.3f * c[globalId]; 
}

kernel void triad16(
    global TYPE16 *restrict a,
    global const TYPE16 *restrict b,
    global const TYPE16 *restrict c)
{
    int globalId = get_global_id(0);
    a[globalId] = b[globalId] + 0.3f * c[globalId]; 
}

#endif