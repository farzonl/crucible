cmake_minimum_required(VERSION 3.24)
project(crucible LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CRUCIBLE_ENABLE_TESTS "Build crucible lit/FileCheck test suite" ON)

# --- LLVM discovery (mirrors Nacelle's search order) --------------------------
if(NOT LLVM_DIR)
  foreach(hint
      "${CMAKE_SOURCE_DIR}/.toolchains/llvm/lib/cmake/llvm"
      "/opt/homebrew/opt/llvm/lib/cmake/llvm"
      "/usr/local/opt/llvm/lib/cmake/llvm"
      "/usr/lib/llvm-24/lib/cmake/llvm"
      "/usr/lib/llvm-23/lib/cmake/llvm"
      "/usr/lib/llvm-22/lib/cmake/llvm"
      "/usr/lib/llvm-21/lib/cmake/llvm")
    if(EXISTS "${hint}/LLVMConfig.cmake")
      set(LLVM_DIR "${hint}")
      break()
    endif()
  endforeach()
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS_LIST})

llvm_map_components_to_libnames(CRUCIBLE_LLVM_LIBS
  Core Object BitReader IRReader BinaryFormat Support TargetParser
  Analysis Passes TransformUtils ScalarOpts)

# Match LLVM conventions: no RTTI / no exceptions unless LLVM has them.
if(NOT LLVM_ENABLE_RTTI)
  set(CRUCIBLE_NO_RTTI_FLAGS -fno-rtti)
endif()
set(CRUCIBLE_COMMON_FLAGS -fno-exceptions ${CRUCIBLE_NO_RTTI_FLAGS})

add_subdirectory(lib)
add_subdirectory(tools/cruciblec)
if(APPLE)
  add_subdirectory(tools/crucible-run)
endif()

if(CRUCIBLE_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
