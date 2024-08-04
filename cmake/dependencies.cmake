# download lib
set(BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)

# spdlog
message(STATUS "Setup spdlog")
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.13.0
  SOURCE_DIR ${BUILD_LIB_DIR}/spdlog
)
FetchContent_MakeAvailable(spdlog)
set(SPDLOG_INCLUDE ${BUILD_LIB_DIR}/spdlog/include)
target_include_directories(${LIB_NAME} PUBLIC ${SPDLOG_INCLUDE})
target_link_libraries(${LIB_NAME} PUBLIC spdlog)

# Catch2
message(STATUS "Setup catch2")
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
  SOURCE_DIR ${BUILD_LIB_DIR}/catch2
)
FetchContent_MakeAvailable(Catch2)
target_include_directories(${TEST_NAME} PUBLIC ${BUILD_LIB_DIR}/catch2/src/)
target_link_libraries(${TEST_NAME} PUBLIC Catch2::Catch2)
