file(GLOB_RECURSE TEST_SRC
    "main.cc"
    "test_*.cc"
    "test_*.cu"
)
target_sources(${TEST_NAME} PRIVATE ${TEST_SRC})
