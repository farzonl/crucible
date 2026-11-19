# Build the suite's unmodified CLI sources with Crucible as their device provider.

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
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/API/Capabilities.cpp"
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
add_library(crucible-offload-device STATIC Device.cpp)
target_include_directories(crucible-offload-device PRIVATE
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/third-party/metal-cpp")
target_link_libraries(crucible-offload-device PUBLIC
  crucible
  offload-suite-support
  "-framework Metal"
  "-framework Foundation"
  "-framework QuartzCore"
)

find_package(PNG REQUIRED)
add_library(offload-suite-image STATIC
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Image/Color.cpp"
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Image/Image.cpp"
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Image/ImageComparators.cpp"
)
target_compile_options(offload-suite-image PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(offload-suite-image PUBLIC offload-suite-support PNG::PNG)

add_executable(imgdiff "${CRUCIBLE_OFFLOAD_SUITE_DIR}/tools/imgdiff/imgdiff.cpp")
target_compile_options(imgdiff PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(imgdiff PRIVATE offload-suite-image)

add_executable(offloader "${CRUCIBLE_OFFLOAD_SUITE_DIR}/tools/offloader/offloader.cpp")
target_link_libraries(offloader PRIVATE crucible-offload-device offload-suite-image)
add_executable(api-query "${CRUCIBLE_OFFLOAD_SUITE_DIR}/tools/api-query/api-query.cpp"
  "${CRUCIBLE_OFFLOAD_SUITE_DIR}/lib/Support/HostInfo.cpp")
target_compile_definitions(api-query PRIVATE OFFLOADTEST_HAVE_SYSCTLBYNAME)
target_link_libraries(api-query PRIVATE crucible-offload-device)
foreach(tool offloader api-query)
  target_compile_options(${tool} PRIVATE ${CRUCIBLE_COMMON_FLAGS})
endforeach()
