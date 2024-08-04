#ifndef SRC_COMMON_UTILS_H_
#define SRC_COMMON_UTILS_H_

#include <source_location>
#include <string>

namespace cupr {
std::string MakeErrorMessage(const std::string_view msg,
                             const std::source_location location = std::source_location::current());
}  // namespace cupr

#endif  // SRC_COMMON_UTILS_H_
