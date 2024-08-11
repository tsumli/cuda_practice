#include <catch2/catch_all.hpp>

#include "common/cuda/exception.h"

TEST_CASE("THROW_IF_FAILED") {
    REQUIRE_NOTHROW(THROW_IF_FAILED(cudaSuccess));
    REQUIRE_THROWS(THROW_IF_FAILED(cudaGetDeviceProperties(nullptr, 0)));
}