#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/exception.h"

__global__ void HelloDevice() { printf("Hello Device!\n"); }

int main() {
    std::cout << "Hello Host!" << std::endl;
    HelloDevice<<<1, 1>>>();
    THROW_IF_FAILED(cudaDeviceSynchronize());
    return 0;
}
