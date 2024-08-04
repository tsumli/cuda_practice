#include <catch2/catch_all.hpp>

#include "common/utils.h"

TEST_CASE("GetFileAndLine") {
    const std::string msg = cupr::MakeErrorMessage("test");
    REQUIRE(msg == "/workspace/test/common/test_utils.cc (line 6): test");
}
