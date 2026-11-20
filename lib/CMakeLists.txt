add_subdirectory(Dialect)

add_library(crucible STATIC
  Compiler.cpp
  Reflection.cpp
  Frontend/ContainerLoader.cpp
  Frontend/DXILUpgrader.cpp
  Frontend/MLIRLoader.cpp
  Frontend/SPIRVLoader.cpp
  Transforms/ResourceLowering.cpp
  Transforms/CFGStructurizer.cpp
  Transforms/DXILToLayer1.cpp
  Backend/MSLEmitter.cpp
  Backend/SPIRVEmitter.cpp
)

target_include_directories(crucible PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)
target_compile_options(crucible PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(crucible PUBLIC crucible_dialect ${CRUCIBLE_MLIR_LIBS} ${CRUCIBLE_LLVM_LIBS})
