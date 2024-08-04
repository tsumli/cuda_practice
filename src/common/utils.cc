#include "common/utils.h"

namespace cupr {
std::string MakeErrorMessage(const std::string_view msg, const std::source_location location) {
    const auto file = std::string(location.file_name());
    const auto line = std::to_string(location.line());
    return file + " (line " + line + ")" + ": " + std::string(msg);
}
}  // namespace cupr
