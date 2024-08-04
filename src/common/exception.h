#ifndef SRC_COMMON_EXCEPTION_H_
#define SRC_COMMON_EXCEPTION_H_

#include <cuda_runtime.h>

#include <sstream>

namespace cupr {
template <typename F, typename N>
void ThrowIfFailed(const ::cudaError_t e, F&& f, N&& n) {
    if (e != cudaSuccess) {
        std::ostringstream ss;
        ss << cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error{ss.str()};
    }
}
#define THROW_IF_FAILED(e) (cupr::ThrowIfFailed(e, __FILE__, __LINE__))
}  // namespace cupr

#endif  // SRC_COMMON_EXCEPTION_H_
