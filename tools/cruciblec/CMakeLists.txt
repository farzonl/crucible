add_executable(cruciblec main.cpp)
target_compile_options(cruciblec PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(cruciblec PRIVATE crucible)
