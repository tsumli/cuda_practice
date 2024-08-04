set(executable_name 01-helloworld)
add_executable(${executable_name})
target_link_libraries(${executable_name} PRIVATE ${LIB_NAME})

target_sources(${executable_name} PRIVATE main.cc)