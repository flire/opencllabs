#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <algorithm>

#include <memory>

using namespace std;

const size_t BLOCK_SIZE = 256;

struct OpenCLMeta
{
    unique_ptr<cl::Context> context;
    unique_ptr<cl::CommandQueue> queue;
    unique_ptr<cl::Program> program;
};

unique_ptr<OpenCLMeta> initOpenCL()
{
    unique_ptr<OpenCLMeta> result = make_unique<OpenCLMeta>();

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    result->context = make_unique<cl::Context>(devices);

    // create command queue
    result->queue = make_unique<cl::CommandQueue>(*(result->context), devices[0], CL_QUEUE_PROFILING_ENABLE);

    // load opencl source
    ifstream cl_file("scan.cl");
    string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(cl_string.c_str(),
        cl_string.length() + 1));

    // create program
    result->program = make_unique<cl::Program>(*(result->context), source);

    // compile opencl source
    result->program->build(devices);

    return result;
}

typedef cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg, cl::LocalSpaceArg> ScanKernel;
typedef cl::make_kernel<cl::Buffer &, cl::Buffer &> PropagateKernel;

vector<float> getPrefixSum(vector<float> &arr, ScanKernel &scanner, PropagateKernel &propagator, unique_ptr<OpenCLMeta> &opencl)
{
    int boundarySumsCount = arr.size() / BLOCK_SIZE;
    if (boundarySumsCount > BLOCK_SIZE && boundarySumsCount % BLOCK_SIZE != 0)
    {
        boundarySumsCount += BLOCK_SIZE - (boundarySumsCount % BLOCK_SIZE);
    }
    if (boundarySumsCount == 0)
    {
        boundarySumsCount = 1;
    }

    vector<float> boundarySums(boundarySumsCount, 0.);

    cl::Buffer input_buffer(*(opencl->context), CL_MEM_READ_ONLY, sizeof(float) * arr.size());
    cl::Buffer output_buffer(*(opencl->context), CL_MEM_READ_WRITE, sizeof(float) * arr.size());
    cl::Buffer boundary_buffer(*(opencl->context), CL_MEM_READ_WRITE, sizeof(float) * boundarySumsCount);

    opencl->queue->enqueueWriteBuffer(input_buffer, CL_TRUE, 0, sizeof(float) * arr.size(), arr.data());

    cl::EnqueueArgs scan_args(*opencl->queue, cl::NullRange, cl::NDRange(arr.size()), cl::NDRange(min(BLOCK_SIZE, arr.size())));
    scanner(scan_args, input_buffer, output_buffer, boundary_buffer, cl::Local(sizeof(float) * min(BLOCK_SIZE, arr.size())), cl::Local(sizeof(float) * min(BLOCK_SIZE, arr.size())));

    if (arr.size() > BLOCK_SIZE) //than boundary sums are needed to be processed as well
    {
        opencl->queue->enqueueReadBuffer(boundary_buffer, CL_TRUE, 0, sizeof(float) * boundarySums.size(), boundarySums.data());
        boundarySums = getPrefixSum(boundarySums, scanner, propagator, opencl);
        opencl->queue->enqueueWriteBuffer(boundary_buffer, CL_TRUE, 0, sizeof(float) * boundarySums.size(), boundarySums.data());

        cl::EnqueueArgs add_args(*opencl->queue, cl::NullRange, cl::NDRange(arr.size()), cl::NDRange(BLOCK_SIZE));
        propagator(add_args, output_buffer, boundary_buffer);
    }

    vector<float> output_vector(arr.size(), 0.);
    opencl->queue->enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * output_vector.size(), output_vector.data());
    return output_vector;
}

int main()
{

    try {

        auto opencl = initOpenCL();

        ifstream in("input.txt");
        int N;
        in >> N;

        int block_size = 256;

        int array_size = N;
        if (N % block_size != 0)
        {
            array_size += block_size - (N % block_size);
        }

        vector<float> input_vector(array_size, 0.);
        for (int i = 0; i < N; ++i)
        {
            in >> input_vector[i];
        }

        cl::Kernel scan_kernel(*opencl->program, "scan_hillis_steele");
        ScanKernel scan_functor(scan_kernel);

        cl::Kernel add_kernel(*opencl->program, "propagate_boundaries");
        PropagateKernel add_functor(add_kernel);

        auto output_vector = getPrefixSum(input_vector, scan_functor, add_functor, opencl);


        ofstream out("output.txt");
        out << fixed << setprecision(3);
        for (int i = 0; i < N; ++i)
        {
            out << output_vector[i] << " ";
        }
        out << endl;
    }
    catch (cl::Error e)
    {
        cout << endl << e.what() << " : " << e.err() << endl;
    }

    return 0;
}