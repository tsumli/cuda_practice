/**
 * @file main.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <indicators/progress_bar.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "common/cuda/exception.h"
#include "common/cuda/pointer.h"
#include "common/cuda/utils.h"
#include "common/progress_bar.h"
#include "common/timer.h"

template <uint WarpSize>
__global__ void reduction_shared_mem(int *const output, const int *const input, const int size) {
    const auto tid = threadIdx.x;
    const auto num_threads = blockDim.x;
    const auto num_blocks = gridDim.x;

    // Compute sum of elements in the block
    auto sum{0};
    for (auto i = tid; i < size; i += num_threads * num_blocks) {
        sum += input[i];
    }

    // Store the sum in shared memory
    __shared__ int shared_sum[WarpSize];
    shared_sum[tid] = sum;

    __syncthreads();

    for (auto stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_sum[0];
    }
}

__global__ void reduction_shuffle(int *const output, const int *const input, const int size) {
    const auto tid = threadIdx.x;
    const auto num_threads = blockDim.x;
    const auto num_blocks = gridDim.x;

    auto sum{0};
    for (auto i = tid; i < size; i += num_threads * num_blocks) {
        sum += input[i];
    }

    for (auto stride = num_threads / 2; stride > 0; stride /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
    }

    if (tid == 0) {
        *output = sum;
    }
}

template <typename Function>
std::optional<std::uint32_t> TestReductionKernel(Function fn, const std::size_t kInputSize) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-100, 100);

    // Generate random input data
    auto input_host = std::vector<int>{};
    input_host.reserve(kInputSize);
    auto ans_expected{0};
    for (std::size_t i = 0; i < kInputSize; i++) {
        const auto num{dis(gen)};
        input_host.emplace_back(num);
        ans_expected += num;
    }
    auto input_device = cupr::cuda::make_unique<int[]>(kInputSize);
    cupr::cuda::CopyToDevice(input_device.get(), input_host.data(), kInputSize);

    // Allocate memory for output
    auto output_device = cupr::cuda::make_unique<int>();

    // Call reduction kernel here
    std::uint32_t duration;
    {
        const auto timer = cupr::ScopedTimer(duration);
        fn<<<1, 32>>>(output_device.get(), input_device.get(), kInputSize);
        THROW_IF_FAILED(cudaDeviceSynchronize());
    }

    const auto output_host = cupr::cuda::CopyFromDevice(output_device.get());

    if (ans_expected != output_host) {
        std::cerr << "Test failed" << std::endl;
        std::cerr << "Expected: " << ans_expected << std::endl;
        std::cerr << "Computed by kernel: " << output_host << std::endl;
        return std::nullopt;
    }
    return duration;
}

double compute_average(const std::vector<uint32_t> &values) {
    if (values.empty()) {
        return 0.0;  // Handle empty vector case
    }

    uint64_t sum = std::accumulate(values.begin(), values.end(), uint64_t(0));
    return static_cast<double>(sum) / values.size();
}

int main() {
    constexpr std::size_t kInputSize = 10000;
    constexpr std::size_t kTestCount = 1000;
    constexpr std::uint32_t WarpSize = 32;

    // Test reduction kernels (`reduction_shared_mem`)
    {
        std::cout << "Test `reduction_shared_mem` kernel" << std::endl;
        auto bar = cupr::CreateProgressBar("Testing for " + std::to_string(kTestCount) + " times");
        auto durations = std::vector<std::uint32_t>{};
        durations.reserve(kTestCount);
        for (std::size_t test_i = 0; test_i < kTestCount; test_i++) {
            bar.set_progress(static_cast<double>(test_i + 1) * 100 / kTestCount);
            const auto duration = TestReductionKernel(reduction_shared_mem<WarpSize>, kInputSize);
            if (!duration) {
                std::cout << "Test failed" << std::endl;
                return EXIT_FAILURE;
            }
            durations.emplace_back(duration.value());
        }
        std::cout << "Duration average: " << compute_average(durations) << " [us]" << std::endl;
        std::cout << "All tests passed" << std::endl;
    }

    // Test reduction kernels (`reduction_shuffle`)
    {
        std::cout << "Test `reduction_shuffle` kernel" << std::endl;
        auto bar = cupr::CreateProgressBar("Testing for " + std::to_string(kTestCount) + " times");
        auto durations = std::vector<std::uint32_t>{};
        durations.reserve(kTestCount);
        for (std::size_t test_i = 0; test_i < kTestCount; test_i++) {
            bar.set_progress(static_cast<double>(test_i + 1) * 100 / kTestCount);
            const auto duration = TestReductionKernel(reduction_shared_mem<WarpSize>, kInputSize);
            if (!duration) {
                std::cout << "Test failed" << std::endl;
                return EXIT_FAILURE;
            }
            durations.emplace_back(duration.value());
        }
        std::cout << "Duration average: " << compute_average(durations) << " [us]" << std::endl;
        std::cout << "All tests passed" << std::endl;
    }
    return EXIT_SUCCESS;
}