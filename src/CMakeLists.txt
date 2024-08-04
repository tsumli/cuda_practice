set(LIB_SRC
    common/utils.h
)
target_sources(${LIB_NAME} PRIVATE ${LIB_SRC})

add_subdirectory(01-helloworld)