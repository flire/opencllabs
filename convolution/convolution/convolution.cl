__kernel void gpu_convolution_gmem(__global float * input, __global float * mask, 
                                   __global float * output, int mask_width, int width)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= width || j >= width)
        return;

    float result = 0;

    for (int mask_row = 0; mask_row < mask_width; ++mask_row) 
    {
        for (int mask_column = 0; mask_column < mask_width; ++mask_column)
        {
            int input_i = i + mask_row - mask_width / 2;
            int input_j = j + mask_column - mask_width / 2;
            if (0 <= input_i && input_i < width && 0 <= input_j && input_j < width)
            {
                result += input[input_i * width + input_j] * mask[mask_row * mask_width + mask_column];
            }
        }
    }

    output[i * width + j] = result;
}

