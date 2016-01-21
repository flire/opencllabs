#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

using namespace std;

int main()
{
   vector<cl::Platform> platforms;
   vector<cl::Device> devices;
   vector<cl::Kernel> kernels;

   try {

       // create platform
       cl::Platform::get(&platforms);
       platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

       // create context
       cl::Context context(devices);

       // create command queue
       cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

       // load opencl source
       ifstream cl_file("convolution.cl");
       string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
       cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

       // create program
       cl::Program program(context, source);

       // compile opencl source
       program.build(devices);

       fstream in("input.txt");

       int m, n;

       in >> n >> m;

       vector<float> A(n * n, 1.);

       for (int row = 0; row < n; row++)
       {
           for (int column = 0; column < n; column++) 
           {
               float element;
               in >> element;
               A[row * n + column] = element;
           }
       }

       vector<float> B(m * m, 1.);

       for (int row = 0; row < m; row++)
       {
           for (int column = 0; column < m; column++)
           {
               float element;
               in >> element;
               B[row * m + column] = element;
           }
       }

       // allocate device buffer to hold message
       cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * n * n);
       cl::Buffer dev_mask(context, CL_MEM_READ_ONLY, sizeof(float) * m * m);
       cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

       // copy from cpu to gpu
       queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n * n, &A[0]);
       queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(float) * m * m, &B[0]);

       // load named kernel from opencl source
       cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
       cl::make_kernel<cl::Buffer &, cl::Buffer&, cl::Buffer&, int, int> kernel_functor(kernel_gmem);
       int size = n % 16 == 0 ? n : n + (16 - n % 16);
       cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(size, size), cl::NDRange(16, 16));
       kernel_functor(eargs, dev_input, dev_mask, dev_output, m, n);

       vector<float> C(n * n, 1.);

       queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n * n, &C[0]);

       ofstream out("output.txt");

       out << fixed << setprecision(3);

       for (size_t i = 0; i < n; ++i)
       {
           for (size_t j = 0; j < n; ++j)
           {
               out << C[i * n + j] << " ";
           }
           out << endl;
       }
   }
   catch (cl::Error e)
   {
      cout << endl << e.what() << " : " << e.err() << endl;
   }

   return 0;
}