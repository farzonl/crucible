add_library(crucible STATIC
  Compiler.cpp
  Reflection.cpp
  Frontend/ContainerLoader.cpp
  Frontend/DXILUpgrader.cpp
  Transforms/ResourceLowering.cpp
  Backend/IntrinsicMap.cpp
  Backend/MSLEmitter.cpp
)

target_include_directories(crucible PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_compile_options(crucible PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(crucible PUBLIC ${CRUCIBLE_LLVM_LIBS})
