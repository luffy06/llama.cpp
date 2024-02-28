#include <iostream>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast.h>

#define MULTILINE_QUOTE(...) #__VA_ARGS__
static std::string kernel_source = MULTILINE_QUOTE(
    __kernel void softmax(__global const float *input, __global float *output, const int input_size, __local float *local_max, __local float *local_sum) {
        int global_id = get_global_id(0);
        int local_id = get_local_id(0);
        int group_id = get_group_id(0);
        int local_size = get_local_size(0);
        int group_size = get_num_groups(0);
        int local_start = group_id * local_size;
        int local_end = local_start + local_size;
        int i;
        float max = -INFINITY;
        float sum = 0.0f;
        for (i = local_start; i < local_end; i++) {
            max = fmax(max, input[i]);
        }
        local_max[local_id] = max;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (i = 0; i < local_size; i++) {
            max = fmax(max, local_max[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (i = local_start; i < local_end; i++) {
            sum += exp(input[i] - max);
        }
        local_sum[local_id] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        sum = 0.0f;
        for (i = 0; i < local_size; i++) {
            sum += local_sum[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (i = local_start; i < local_end; i++) {
            output[i] = exp(input[i] - max) / sum;
        }
    }
);

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

enum { NPLAT = 1, NDEV = 1 };

std::string generate_kernel_source();

int main() {
    cl_int ret;
    // init environment
    cl_platform_id platform_id      = NULL;
    cl_device_id device_id          = NULL;
    cl_context context              = NULL;
    cl_command_queue command_queue  = NULL;
    cl_program program              = NULL;

    {
        // get platform and device ids
        CL_CHECK(clGetPlatformIDs(NPLAT, &platform_id, NULL));
        CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, NDEV, &device_id, NULL));

        // create context and command queue
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);
    }

    // init kernel
    cl_kernel kernel                = NULL;

    {
        // load kernel source codes
        const char* program_buffer = generate_kernel_source().c_str();
        size_t program_size = strlen(program_buffer);
        
        // create program with source codes
        program = clCreateProgramWithSource(context, NDEV, (const char**)&program_buffer, (const size_t *)&program_size, &ret);
        ret = clBuildProgram(program, NDEV, &device_id, NULL, NULL, NULL);
        kernel = clCreateKernel(program, "softmax", &ret);
    }

    // generate input data
    size_t input_size = 100;
    std::vector<float> input(input_size);
    std::vector<float> output(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        input[i] = i;
    }

    // create memory object in opencl devices
    cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_size, input.data(), &ret);
    cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * input_size, NULL, &ret);
    
    size_t local_work_size = 256;
    size_t global_work_size = ((input_size + local_work_size - 1) / local_work_size) * local_work_size;
    size_t local_memory_size = local_work_size * sizeof(float);

    // set kernel arguments
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem_obj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem_obj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(size_t), &input_size));
    CL_CHECK(clSetKernelArg(kernel, 3, local_memory_size, NULL));
    CL_CHECK(clSetKernelArg(kernel, 4, local_memory_size, NULL));

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clFinish(command_queue);            
    ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0, input_size * sizeof(float), output.data(), 0, NULL, NULL);

    // clear kernels
    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(input_mem_obj);
    ret = clReleaseMemObject(output_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // print results
    for (size_t i = 0; i < input_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

std::string generate_kernel_source() {
    std::stringstream src;
    src << kernel_source << "\n";
    return src.str();
}
