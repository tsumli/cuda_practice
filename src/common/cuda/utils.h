#ifndef SRC_COMMON_CUDA_UTILS_H_
#define SRC_COMMON_CUDA_UTILS_H_

#include <cuda_runtime.h>

#include <vector>

#include "common/cuda/exception.h"

namespace cupr::cuda {
template <typename T>
T GetValueFromDevice(const T* const value_device) {
    T value;
    THROW_IF_FAILED(cudaMemcpy(&value, value_device, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
}

template <typename T>
std::vector<T> GetVectorFromDevice(const T* const value_device, const size_t size) {
    std::vector<T> value(size);
    THROW_IF_FAILED(
        cudaMemcpy(value.data(), value_device, size * sizeof(T), cudaMemcpyDeviceToHost));
    return value;
}

template <typename T>
void CopyToDevice(T* value_device, const T* const value_host, const std::size_t size) {
    THROW_IF_FAILED(cudaMemcpy(value_device, value_host, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template <typename T>
constexpr T DivUp(T value, T s) {
    return (value + s - 1) / s;
}
}  // namespace cupr::cuda

#endif  // SRC_COMMON_CUDA_UTILS_H_
