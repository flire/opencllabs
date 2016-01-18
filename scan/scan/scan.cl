#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global float * input, __global float * output, __global float *boundaries, __local float * a, __local float * b)
{
    uint lid = get_local_id(0);
    uint grid = get_group_id(0);
    uint block_size = get_local_size(0);
    uint effective_index = lid + grid * block_size;
 
    a[lid] = b[lid] = input[effective_index];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }

    output[effective_index] = a[lid];
    if (lid == block_size - 1) boundaries[grid] = a[lid];
}

__kernel void propagate_boundaries(__global float * input, __global float * boundaries)
{
    uint lid = get_local_id(0);
    uint grid = get_group_id(0);
    uint block_size = get_local_size(0);

    if (grid > 0)
    {
        input[lid + grid * block_size] += boundaries[grid - 1];
    }
}