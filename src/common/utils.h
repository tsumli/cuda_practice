#ifndef SRC_COMMON_UTILS_H_
#define SRC_COMMON_UTILS_H_

#include <indicators/progress_bar.hpp>

namespace cupr {
indicators::ProgressBar CreateProgressBar(const std::string_view postfix,
                                          const std::uint32_t bar_width = 50);

}  // namespace cupr

#endif  // SRC_COMMON_UTILS_H_
