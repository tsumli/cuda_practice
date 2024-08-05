/**
 * @file main.cu
 * @brief https://developer.nvidia.com/blog/cooperative-groups/
 */

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"
#include "common/exception.h"

namespace cg = cooperative_groups;

/**
 * @brief Sum elements of input array
 *
 * @param input Input array
 * @param n Number of elements
 * @return Sum of elements
 */
__device__ int thread_sum(int *input, int n) {
    int sum = 0;
    cg::thread_block block = cg::this_thread_block();
    printf("blockDim.x: %d\n", blockDim.x);
    printf("block.size: %d\n", block.size());
    for (int i = block.group_index().x * block.size() + threadIdx.x; i < n / 4;
         i += blockDim.x * gridDim.x) {
        int4 in = ((int4 *)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    cg::coalesced_group active = cg::coalesced_threads();

    return sum;
}

template <int tile_sz>
__device__ int reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g, int val) {
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;  // note: only thread 0 will return full sum
}

template <int tile_sz>
__global__ void sum_kernel_tile_shfl(int *sum, int *input, int n) {
    int my_sum = thread_sum(input, n);

    auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    int tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, my_sum);

    if (tile.thread_rank() == 0) {
        atomicAdd(sum, tile_sum);
    }
}

int main() {
    const auto sum_host = 0;
    auto sum_device = cupr::cuda::make_unique<int>();
    cupr::cuda::CopyToDevice(sum_device.get(), &sum_host, 1);

    const auto input_host = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto input_device = cupr::cuda::make_unique<int[]>(input_host.size());
    cupr::cuda::CopyToDevice(input_device.get(), input_host.data(), input_host.size());

    const std::uint32_t block_dim =
        cupr::cuda::DivUp(input_host.size(), std::size_t{32}) * input_host.size();
    sum_kernel_tile_shfl<32>
        <<<1, block_dim>>>(sum_device.get(), input_device.get(), input_host.size());
    THROW_IF_FAILED(cudaDeviceSynchronize());

    const auto sum = cupr::cuda::GetValueFromDevice(sum_device.get());
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
