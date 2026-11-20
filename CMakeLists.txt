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

if(NOT MLIR_DIR)
  get_filename_component(CRUCIBLE_LLVM_PREFIX "${LLVM_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${CRUCIBLE_LLVM_PREFIX}/lib/cmake/mlir/MLIRConfig.cmake")
    set(MLIR_DIR "${CRUCIBLE_LLVM_PREFIX}/lib/cmake/mlir")
  endif()
endif()
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR at ${MLIR_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Prefer the monolithic LLVM dylib when present (Homebrew ships one and its
# MLIR static archives depend on it); mixing it with static component
# archives duplicates LLVM globals and crashes at runtime.
if(TARGET LLVM)
  set(CRUCIBLE_LLVM_LIBS LLVM)
else()
  llvm_map_components_to_libnames(CRUCIBLE_LLVM_LIBS
    Core Object BitReader IRReader BinaryFormat Support TargetParser
    Analysis Passes TransformUtils ScalarOpts
    SPIRVCodeGen SPIRVDesc SPIRVInfo CodeGen Target MC AsmPrinter)
endif()

# MLIR is used to lift the unstructured DXIL CFG into structured scf.if /
# scf.while regions (via LiftControlFlowToSCF) that the MSL emitter walks.
set(CRUCIBLE_MLIR_LIBS
  MLIRTargetLLVMIRImport
  MLIRLLVMIRToLLVMTranslation
  MLIRSPIRVDeserialization
  MLIRSPIRVToLLVM
  MLIRSPIRVDialect
  MLIRGPUDialect
  MLIRMathDialect
  MLIRControlFlowToSCF
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRUBDialect
  MLIRDLTIDialect
  MLIRTransforms
  MLIRPass
  MLIRParser
  MLIRIR)

# Match LLVM conventions: no RTTI / no exceptions unless LLVM has them.
if(NOT LLVM_ENABLE_RTTI)
  set(CRUCIBLE_NO_RTTI_FLAGS -fno-rtti)
endif()
set(CRUCIBLE_COMMON_FLAGS -fno-exceptions ${CRUCIBLE_NO_RTTI_FLAGS})
if(UNIX AND NOT APPLE)
  # Keep Crucible and LLVM/MLIR template instantiations hidden from the ELF
  # dynamic symbol table so Vulkan ICDs (e.g., Mesa Lavapipe linked against a
  # different system libLLVM.so) cannot interpose on them at runtime.
  list(APPEND CRUCIBLE_COMMON_FLAGS -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

add_subdirectory(lib)
add_subdirectory(tools/cruciblec)
if(APPLE)
  add_subdirectory(tools/crucible-run)
endif()

# Optional Vulkan / MoltenVK E2E compute runner (crucible-vk-run).
find_path(CRUCIBLE_VULKAN_INCLUDE_DIR vulkan/vulkan.h
  HINTS /usr/local/include /opt/homebrew/include)
find_library(CRUCIBLE_VULKAN_LIB NAMES vulkan vulkan-1 libMoltenVK.dylib
  HINTS /opt/homebrew/lib /usr/local/lib)
if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  message(STATUS "Found Vulkan/MoltenVK: ${CRUCIBLE_VULKAN_LIB}")
  add_subdirectory(tools/crucible-vk-run)
endif()

# llvm/offload-test-suite integration: builds the upstream CLI tools with
# Crucible's Metal execution provider, plus a check-crucible-offload lit target.
# over the suite's test tree.
set(CRUCIBLE_OFFLOAD_SUITE_DIR "" CACHE PATH
    "Path to an llvm/offload-test-suite checkout")
if(NOT CRUCIBLE_OFFLOAD_SUITE_DIR)
  foreach(candidate
      "${CMAKE_SOURCE_DIR}/third_party/offload-test-suite"
      "${CMAKE_SOURCE_DIR}/../offload-test-suite")
    if(EXISTS "${candidate}/include/Support/Pipeline.h")
      set(CRUCIBLE_OFFLOAD_SUITE_DIR "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(CRUCIBLE_OFFLOAD_SUITE_DIR)
  message(STATUS "offload-test-suite: ${CRUCIBLE_OFFLOAD_SUITE_DIR}")
  add_subdirectory(tools/crucible-offload)
endif()

set(CRUCIBLE_GOLDENIMAGE_DIR "" CACHE PATH
    "Path to an llvm/offload-golden-images checkout")
if(NOT CRUCIBLE_GOLDENIMAGE_DIR)
  foreach(candidate
      "${CMAKE_SOURCE_DIR}/third_party/offload-golden-images"
      "${CMAKE_SOURCE_DIR}/../offload-golden-images"
      "${CMAKE_BINARY_DIR}/golden-images")
    if(EXISTS "${candidate}")
      set(CRUCIBLE_GOLDENIMAGE_DIR "${candidate}")
      break()
    endif()
  endforeach()
endif()

if(CRUCIBLE_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
