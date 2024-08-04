#ifndef SRC_COMMON_CUDA_POINTER_H_
#define SRC_COMMON_CUDA_POINTER_H_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <memory>

#include "common/exception.h"

namespace cupr::cuda {

struct deleter {
    void operator()(void* p) const { THROW_IF_FAILED(cudaFree(p)); }
};
template <typename T>
using unique_ptr = std::unique_ptr<T, deleter>;

/**
 * @brief Make a unique pointer of type T
 * @code auto array = cuda::make_unique<std::vector<float>>(n); @endcode
 * @param n The size of the array
 * @return cuda::unique_ptr<T> A unique pointer of type T
 */
template <typename T>
typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(
    const std::size_t n) {
    using U = typename std::remove_extent<T>::type;
    U* p;
    THROW_IF_FAILED(cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
    return cuda::unique_ptr<T>{p};
}

/**
 * @brief Make a unique pointer of type T
 * @code auto value = cuda::make_unique<float>(); @endcode
 */
template <typename T>
cuda::unique_ptr<T> make_unique() {
    T* p;
    THROW_IF_FAILED(cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)));
    return cuda::unique_ptr<T>{p};
}

}  // namespace cupr::cuda

#endif  // SRC_COMMON_CUDA_POINTER_H_
