#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

__global__ void HelloFromGPU() { printf("Hello world from GPU!\n"); }

int main() {
    std::cout << "Hello world from Host!" << std::endl;
    HelloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
