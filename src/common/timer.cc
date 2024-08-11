#include "timer.h"

namespace cupr {
ScopedTimer::ScopedTimer(std::uint32_t& duration)
    : duration_{duration}, start_time_{std::chrono::high_resolution_clock::now()} {}

ScopedTimer::~ScopedTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
    duration_ = duration.count();
}
}  // namespace cupr