# project settings
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(cuda_practice LANGUAGES CXX CUDA)
set(LIB_NAME ${PROJECT_NAME}-lib)

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

# cuda settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -ccbin /usr/local/bin/g++ --allow-unsupported-compiler --cuda-gpu-arch=sm_89)

# project settings and add targets
add_library(${LIB_NAME})
target_compile_options(${LIB_NAME} PRIVATE -Wno-missing-field-initializers)
set(CUDA_INCLUDE /usr/local/cuda/include/)
target_include_directories(${LIB_NAME} PUBLIC ${CUDA_INCLUDE})

# add modules
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED True)

# dependencies
include(cmake/dependencies.cmake)

# subdirectory
set(SRC_DIR "./src")
add_subdirectory(${SRC_DIR})