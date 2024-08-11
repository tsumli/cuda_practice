/**
 * @file main.cu
 * @ref https://developer.nvidia.com/blog/cooperative-groups/
 */

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/cuda/exception.h"

namespace cg = cooperative_groups;

__global__ void my_kernel() {
    cg::thread_block block = cg::this_thread_block();

    dim3 dim_threads = block.dim_threads();
    uint num_threads = block.num_threads();
    uint get_type = block.get_type();
    dim3 group_dim = block.group_dim();
    dim3 group_index = block.group_index();
    dim3 thread_index = block.thread_index();
    uint size = block.size();
    uint thread_rank = block.thread_rank();

    printf("dim_threads: (%d, %d, %d)\n", dim_threads.x, dim_threads.y, dim_threads.z);
    // printf("num_threads: %d\n", num_threads);
    // printf("get_type: %d\n", get_type);
    // printf("group_dim: (%d, %d, %d)\n", group_dim.x, group_dim.y, group_dim.z);
    // printf("group_index: (%d, %d, %d)\n", group_index.x, group_index.y, group_index.z);
    // printf("thread_index: (%d, %d, %d)\n", thread_index.x, thread_index.y, thread_index.z);
    // printf("size: %d\n", size);
    // printf("thread_rank: %d\n", thread_rank);
}

int main() {
    my_kernel<<<dim3{2, 3, 4}, dim3{5, 6, 7}>>>();

    THROW_IF_FAILED(cudaDeviceSynchronize());
    return EXIT_SUCCESS;
}
