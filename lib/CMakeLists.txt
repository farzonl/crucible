add_library(crucible STATIC
  Compiler.cpp
  Reflection.cpp
  Frontend/ContainerLoader.cpp
  Frontend/DXILUpgrader.cpp
  Frontend/SPIRVLoader.cpp
  Transforms/ResourceLowering.cpp
  Transforms/CFGStructurizer.cpp
  Backend/IntrinsicMap.cpp
  Backend/MSLEmitter.cpp
  Backend/SPIRVEmitter.cpp
)

target_include_directories(crucible PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_compile_options(crucible PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(crucible PUBLIC ${CRUCIBLE_MLIR_LIBS} ${CRUCIBLE_LLVM_LIBS})
