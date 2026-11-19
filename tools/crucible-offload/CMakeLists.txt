# Build the suite's unmodified CLI sources with Crucible as their device provider.

# The suite's headers expect a build-generated Config.h; crucible uses none
# of the suite's device backends, so generate a minimal one.
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/suite-config/Config.h"
"#ifndef OFFLOADTEST_CONFIG_H
#define OFFLOADTEST_CONFIG_H
#define APPLE 1
#endif
")
else()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/suite-config/Config.h"
"#ifndef OFFLOADTEST_CONFIG_H
#define OFFLOADTEST_CONFIG_H
#define LINUX 1
#endif
")
endif()

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

set(CRUCIBLE_OFFLOAD_SOURCES
  Common.cpp
  DeviceInit.cpp
)
if(APPLE)
  list(APPEND CRUCIBLE_OFFLOAD_SOURCES MetalDevice.cpp)
endif()
if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  list(APPEND CRUCIBLE_OFFLOAD_SOURCES VulkanDevice.cpp)
endif()

add_library(crucible-offload-device STATIC ${CRUCIBLE_OFFLOAD_SOURCES})
target_link_libraries(crucible-offload-device PUBLIC
  crucible
  offload-suite-support
)
target_compile_options(crucible-offload-device PRIVATE ${CRUCIBLE_COMMON_FLAGS})

if(APPLE)
  target_compile_definitions(crucible-offload-device PRIVATE CRUCIBLE_HAS_METAL_OFFLOAD=1)
  target_include_directories(crucible-offload-device PRIVATE
    "${CRUCIBLE_OFFLOAD_SUITE_DIR}/third-party/metal-cpp")
  target_link_libraries(crucible-offload-device PUBLIC
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
  )
endif()

if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  target_compile_definitions(crucible-offload-device PRIVATE CRUCIBLE_HAS_VULKAN_OFFLOAD=1)
  target_include_directories(crucible-offload-device PRIVATE ${CRUCIBLE_VULKAN_INCLUDE_DIR})
  target_link_libraries(crucible-offload-device PUBLIC ${CRUCIBLE_VULKAN_LIB})
endif()

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
if(APPLE)
  target_compile_definitions(api-query PRIVATE OFFLOADTEST_HAVE_SYSCTLBYNAME)
endif()
target_link_libraries(api-query PRIVATE crucible-offload-device)
foreach(tool offloader api-query)
  target_compile_options(${tool} PRIVATE ${CRUCIBLE_COMMON_FLAGS})
  if(APPLE AND CRUCIBLE_VULKAN_LIB)
    get_filename_component(_vk_lib_dir "${CRUCIBLE_VULKAN_LIB}" DIRECTORY)
    set_target_properties(${tool} PROPERTIES BUILD_RPATH "${_vk_lib_dir}")
  endif()
endforeach()
