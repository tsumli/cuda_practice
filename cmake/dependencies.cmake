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