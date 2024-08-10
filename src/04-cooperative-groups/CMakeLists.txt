get_filename_component(executable_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_executable(${executable_name})
target_link_libraries(${executable_name} PRIVATE ${LIB_NAME})

target_sources(${executable_name} PRIVATE main.cu)
