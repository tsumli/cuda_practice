/**
 * @file main.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/barrier>
#include <iostream>

#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"
#include "common/exception.h"

__global__ void example_kernel(char* dst, char* src) {
    cuda::barrier<cuda::thread_scope_system> bar;
    init(&bar, 1);

    cuda::memcpy_async(dst, src, 1, bar);
    cuda::memcpy_async(dst + 1, src + 8, 1, bar);

    bar.arrive_and_wait();
}

int main() {
    const auto src_host =
        std::vector<char>{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};
    auto src_device = cupr::cuda::make_unique<char[]>(src_host.size());
    cupr::cuda::CopyToDevice(src_device.get(), src_host.data(), src_host.size());
    auto dst_device = cupr::cuda::make_unique<char[]>(src_host.size());

    example_kernel<<<1, 2>>>(dst_device.get(), src_device.get());
    THROW_IF_FAILED(cudaDeviceSynchronize());

    const auto dst_host = cupr::cuda::GetVectorFromDevice(dst_device.get(), src_host.size());
    for (auto i = 0; i < src_host.size(); i++) {
        std::cout << src_host[i] << " -> " << dst_host[i] << std::endl;
    }

    return EXIT_SUCCESS;
}
