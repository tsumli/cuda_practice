set(LIB_SRC
    common/utils.hpp
)
target_sources(${LIB_NAME} PRIVATE ${LIB_SRC})

add_subdirectory(01-helloworld)