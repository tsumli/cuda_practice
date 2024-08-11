#ifndef SRC_COMMON_TIMER_H_
#define SRC_COMMON_TIMER_H_

#include <chrono>

namespace cupr {
class ScopedTimer {
   public:
    /**
     * @brief Construct a new Scoped Timer object
     *
     * @param duration The output duration of the timer in microseconds.
     */
    ScopedTimer(std::uint32_t& duration);
    ~ScopedTimer();

   private:
    std::uint32_t& duration_;
    std::chrono::high_resolution_clock::time_point start_time_;
};
}  // namespace cupr

#endif