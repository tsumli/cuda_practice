/**
 * @file main.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/cuda/exception.h"
#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"

__global__ void add_vector(const int* const a, const int* const b, int* c, clock_t* timer) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    clock_t start = clock();
    c[tid] = a[tid] + b[tid];
    clock_t end = clock();
    if (tid == 0) {
        *timer = end - start;
    }
}

int main() {
    const auto a = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto b = std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17, 18};

    auto a_device = cupr::cuda::make_unique<int[]>(a.size());
    auto b_device = cupr::cuda::make_unique<int[]>(b.size());

    cupr::cuda::CopyToDevice(a_device.get(), a.data(), a.size());
    cupr::cuda::CopyToDevice(b_device.get(), b.data(), b.size());
    auto c_device = cupr::cuda::make_unique<int[]>(a.size());

    auto timer_device = cupr::cuda::make_unique<clock_t>();

    add_vector<<<3, 3>>>(a_device.get(), b_device.get(), c_device.get(), timer_device.get());
    THROW_IF_FAILED(cudaDeviceSynchronize());

    const auto c = cupr::cuda::CopyFromDevice(c_device.get(), a.size());
    for (auto i = 0; i < a.size(); i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    const auto timer = cupr::cuda::CopyFromDevice(timer_device.get());
    std::cout << "Clock:" << timer << std::endl;
    return EXIT_SUCCESS;
}
