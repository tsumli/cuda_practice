#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"
#include "common/exception.h"

__global__ void add_vector(const int* const a, const int* const b, int* c) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    c[tid] = a[tid] + b[tid];
}

int main() {
    const auto a = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto b = std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17, 18};

    auto a_device = cupr::cuda::make_unique<int[]>(a.size());
    auto b_device = cupr::cuda::make_unique<int[]>(b.size());

    cupr::cuda::CopyToDevice(a_device.get(), a.data(), a.size());
    cupr::cuda::CopyToDevice(b_device.get(), b.data(), b.size());
    auto c_device = cupr::cuda::make_unique<int[]>(a.size());

    add_vector<<<3, 3>>>(a_device.get(), b_device.get(), c_device.get());
    THROW_IF_FAILED(cudaDeviceSynchronize());

    const auto c = cupr::cuda::GetVectorFromDevice(c_device.get(), a.size());
    for (auto i = 0; i < a.size(); i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }
    return EXIT_SUCCESS;
}
