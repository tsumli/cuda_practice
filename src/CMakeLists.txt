# Build common library
set(LIB_SRC
    common/timer.cc
    common/progress_bar.cc
    common/cuda/exception.h
    common/cuda/utils.h
    common/cuda/pointer.h
)
target_sources(${LIB_NAME} PRIVATE ${LIB_SRC})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Build examples
add_subdirectory(00-helloworld)
add_subdirectory(01-add-scalar)
add_subdirectory(02-add-vector)
add_subdirectory(03-clock)
add_subdirectory(04-cooperative-groups)
add_subdirectory(05-device)
add_subdirectory(06-reduction)
add_subdirectory(08-smid-block)


