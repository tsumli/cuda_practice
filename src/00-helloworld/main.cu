/**
 * @file main.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/cuda/exception.h"

__global__ void hello_device() { printf("Hello Device!\n"); }

int main() {
    std::cout << "Hello Host!" << std::endl;
    hello_device<<<1, 1>>>();
    THROW_IF_FAILED(cudaDeviceSynchronize());
    return EXIT_SUCCESS;
}
