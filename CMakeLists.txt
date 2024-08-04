# project settings
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(cuda_practice LANGUAGES CXX CUDA)
set(LIB_NAME ${PROJECT_NAME}-lib)
set(TEST_NAME ${PROJECT_NAME}-test)

# defaults
set(CMAKE_C_COMPILER /usr/local/bin/gcc CACHE PATH "")
set(CMAKE_CXX_COMPILER /usr/local/bin/g++ CACHE PATH "")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "")

# cpp settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# cuda settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin /usr/local/bin/g++ -allow-unsupported-compiler)
set(CMAKE_CUDA_FLAGS -allow-unsupported-compiler)

# project settings and add targets
add_library(${LIB_NAME})
target_compile_options(${LIB_NAME} PRIVATE -Wno-missing-field-initializers)
set(CUDA_INCLUDE /usr/local/cuda/include/)
target_include_directories(${LIB_NAME} PUBLIC ${CUDA_INCLUDE})

# test target
add_executable(${TEST_NAME})

# add modules
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED True)

# dependencies
include(cmake/dependencies.cmake)

# subdirectory
set(SRC_DIR "./src")
add_subdirectory(${SRC_DIR})

# test
set(TEST_DIR "./test")
add_subdirectory(${TEST_DIR})
target_link_libraries(${TEST_NAME} PUBLIC ${LIB_NAME})
target_include_directories(${TEST_NAME} PUBLIC ${SRC_DIR})
