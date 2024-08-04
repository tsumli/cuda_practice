set(executable_name 01-add-scalar)
add_executable(${executable_name})
target_link_libraries(${executable_name} PRIVATE ${LIB_NAME})

target_sources(${executable_name} PRIVATE main.cu)
