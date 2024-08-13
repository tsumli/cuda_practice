#include "progress_bar.h"

namespace cupr {
indicators::ProgressBar CreateProgressBar(const std::string_view postfix,
                                          const std::uint32_t bar_width) {
    return indicators::ProgressBar{
        indicators::option::BarWidth{bar_width},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PostfixText{postfix},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::ShowPercentage{true},
    };
}

}  // namespace cupr
