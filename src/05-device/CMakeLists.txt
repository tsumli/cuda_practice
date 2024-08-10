get_filename_component(executable_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_executable(${executable_name})
target_link_libraries(${executable_name} PRIVATE ${LIB_NAME})
set_property(TARGET ${executable_name} PROPERTY CUDA_ARCHITECTURES 89)

target_sources(${executable_name} PRIVATE main.cu)
