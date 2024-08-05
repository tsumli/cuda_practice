#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"
#include "common/exception.h"

__global__ void add(const int a, const int b, int* c) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        *c = a + b;
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 100);

    const int a = dis(gen);
    const int b = dis(gen);
    auto c_device = cupr::cuda::make_unique<int>();

    add<<<1, 1>>>(a, b, c_device.get());
    THROW_IF_FAILED(cudaDeviceSynchronize());

    const auto c = cupr::cuda::GetValueFromDevice(c_device.get());
    std::cout << a << " + " << b << " = " << c << std::endl;
    return 0;
}
