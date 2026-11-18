# crucible-offload: an offloader-compatible runner for llvm/offload-test-suite
# that compiles DXIL with crucible and executes on Metal with zero-overhead
# direct bindings (no metal-shaderconverter, no argument-buffer ABI).

# The suite's headers expect a build-generated Config.h; crucible uses none
# of the suite's device backends, so generate a minimal one.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/suite-config/Config.h"
"#ifndef OFFLOADTEST_CONFIG_H
#define OFFLOADTEST_CONFIG_H
#define APPLE 1
#endif
")

# The suite's YAML pipeline parser and result-verification rules are reused
# directly from its sources.
add_library(offload-suite-support STATIC
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Support/Pipeline.cpp"
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Support/Check.cpp"
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Support/OffloadMigration.cpp"
)
target_include_directories(offload-suite-support PUBLIC
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/suite-config")
target_compile_options(offload-suite-support PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(offload-suite-support PUBLIC ${CRUCIBLE_LLVM_LIBS})

# Pure C++ via the suite's vendored metal-cpp (LLVM headers are not
# Objective-C++ clean).
add_executable(crucible-offload main.cpp)
target_include_directories(crucible-offload PRIVATE
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/third-party/metal-cpp")
target_link_libraries(crucible-offload PRIVATE
  crucible
  offload-suite-support
  "-framework Metal"
  "-framework Foundation"
  "-framework QuartzCore"
)
