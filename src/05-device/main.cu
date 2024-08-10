/**
 * @file main.cu
 * @ref https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

int main() {
    int num_devices;

    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz):" << prop.memoryClockRate << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor
                  << std::endl;
        std::cout << "  Max threads per block dimension: (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1]
                  << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Total global memory (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block (bytes): " << prop.sharedMemPerBlock << std::endl;
        std::cout << "  Memory bus width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << std::endl;
    }
}