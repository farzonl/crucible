# In-process Metal benchmark harness: Crucible (DXIL -> MSL -> Metal) vs.
# Apple's Metal Shader Converter (DXIL -> AIR -> Metal) across the full
# Nacelle benchmark catalog.

find_path(METAL_IRCONVERTER_INCLUDE_DIR
  NAMES metal_irconverter/metal_irconverter.h
  HINTS
    "${METAL_IRCONVERTER_DIR}/include"
    "${CMAKE_SOURCE_DIR}/.sdk/include"
    /usr/local/include
    /opt/homebrew/include
)
find_library(METAL_IRCONVERTER_LIB
  NAMES metalirconverter libmetalirconverter.dylib
  HINTS
    "${METAL_IRCONVERTER_DIR}/lib"
    "${CMAKE_SOURCE_DIR}/.sdk/lib"
    /usr/local/lib
    /opt/homebrew/lib
)

add_executable(crucible-bench
  main.mm
  crucible_runner.mm
  irconverter_runner.mm
)

target_compile_options(crucible-bench PRIVATE
  ${CRUCIBLE_COMMON_FLAGS}
  -fobjc-arc
)

target_include_directories(crucible-bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(crucible-bench PRIVATE
  crucible
  "-framework Metal"
  "-framework Foundation"
)

if(METAL_IRCONVERTER_INCLUDE_DIR AND METAL_IRCONVERTER_LIB)
  message(STATUS "crucible-bench: found metal_irconverter at ${METAL_IRCONVERTER_LIB}")
  target_compile_definitions(crucible-bench PRIVATE CRUCIBLE_HAS_METAL_IRCONVERTER=1)
  target_include_directories(crucible-bench PRIVATE "${METAL_IRCONVERTER_INCLUDE_DIR}")
  target_link_libraries(crucible-bench PRIVATE "${METAL_IRCONVERTER_LIB}")
else()
  message(STATUS "crucible-bench: metal_irconverter not found; building Crucible-only")
endif()
