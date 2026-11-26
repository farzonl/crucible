cmake_minimum_required(VERSION 3.24)
project(crucible VERSION 0.0.1 LANGUAGES C CXX)

include(GNUInstallDirs)

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

find_program(CRUCIBLE_MLIR_TBLGEN
  NAMES mlir-tblgen mlir-tblgen-24 mlir-tblgen-23 mlir-tblgen-22 mlir-tblgen-21
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${MLIR_TOOLS_DIR}" "${CRUCIBLE_LLVM_PREFIX}/bin"
  REQUIRED)
message(STATUS "Found mlir-tblgen at ${CRUCIBLE_MLIR_TBLGEN}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS_LIST})

if("DirectX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(CRUCIBLE_HAS_DIRECTX 1)
  message(STATUS "LLVM DirectX target enabled")
else()
  set(CRUCIBLE_HAS_DIRECTX 0)
  message(STATUS "LLVM DirectX target not available in ${LLVM_DIR}")
endif()

# Prefer the monolithic LLVM dylib when present (Homebrew ships one and its
# MLIR static archives depend on it); mixing it with static component
# archives duplicates LLVM globals and crashes at runtime.
if(TARGET LLVM)
  set(CRUCIBLE_LLVM_LIBS LLVM)
else()
  set(_crucible_llvm_components
    Core Object BitReader IRReader BinaryFormat Support TargetParser
    Analysis Passes TransformUtils ScalarOpts
    SPIRVCodeGen SPIRVDesc SPIRVInfo CodeGen Target MC AsmPrinter)
  if(CRUCIBLE_HAS_DIRECTX)
    list(APPEND _crucible_llvm_components DirectXCodeGen DirectXDesc DirectXInfo)
  endif()
  llvm_map_components_to_libnames(CRUCIBLE_LLVM_LIBS ${_crucible_llvm_components})
endif()

# MLIR is used to lift the unstructured DXIL CFG into structured scf.if /
# scf.while regions (via LiftControlFlowToSCF) that the MSL emitter walks.
set(CRUCIBLE_MLIR_LIBS
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
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
if(MSVC)
  # Strip CMake's default /GR and /EHsc to avoid cl.exe D9025 override warnings.
  string(REGEX REPLACE "(^| ) */GR-? *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "(^| ) */EH[a-z-]+ *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CRUCIBLE_COMMON_FLAGS /EHs-c- /bigobj -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
  if(NOT LLVM_ENABLE_RTTI)
    list(APPEND CRUCIBLE_COMMON_FLAGS /GR-)
  endif()
else()
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
endif()

set(CRUCIBLE_SPIRV_HEADERS_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/third_party/spirv-headers/include")
if(NOT EXISTS "${CRUCIBLE_SPIRV_HEADERS_INCLUDE_DIR}/spirv/unified1/spirv.hpp")
  message(FATAL_ERROR
    "SPIRV-Headers not found at ${CRUCIBLE_SPIRV_HEADERS_INCLUDE_DIR}. "
    "Run: git submodule update --init third_party/spirv-headers")
endif()

add_subdirectory(lib)
add_subdirectory(tools/cruciblec)
if(APPLE)
  add_subdirectory(tools/crucible-bench)
endif()

# Optional Vulkan / MoltenVK / Lavapipe support for crucible-run and
# crucible-offload.
find_path(CRUCIBLE_VULKAN_INCLUDE_DIR vulkan/vulkan.h
  HINTS /usr/local/include /opt/homebrew/include)
find_library(CRUCIBLE_VULKAN_LIB NAMES vulkan vulkan-1 libMoltenVK.dylib
  HINTS /opt/homebrew/lib /usr/local/lib)
if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  message(STATUS "Found Vulkan/MoltenVK: ${CRUCIBLE_VULKAN_LIB}")
endif()

# Optional Nacelle D3D12 software runtime for testing DXIL output on macOS/Linux.
set(CRUCIBLE_NACELLE_DIR "" CACHE PATH
    "Path to extracted Nacelle release directory (bin/, lib/, include/)")
if(NOT CRUCIBLE_NACELLE_DIR)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.toolchains/nacelle/lib")
    set(CRUCIBLE_NACELLE_DIR "${CMAKE_SOURCE_DIR}/.toolchains/nacelle")
  endif()
endif()
if(CRUCIBLE_NACELLE_DIR)
  find_program(NACELLE_COMPILE_EXECUTABLE nacelle-compile
    HINTS "${CRUCIBLE_NACELLE_DIR}/bin" NO_DEFAULT_PATH)
  find_library(CRUCIBLE_NACELLE_LIB NAMES nacelle_d3d12 libnacelle_d3d12.dylib libnacelle_d3d12.so
    HINTS "${CRUCIBLE_NACELLE_DIR}/lib" NO_DEFAULT_PATH)
  if(NACELLE_COMPILE_EXECUTABLE)
    message(STATUS "Found nacelle-compile: ${NACELLE_COMPILE_EXECUTABLE}")
  endif()
  if(CRUCIBLE_NACELLE_LIB)
    message(STATUS "Found Nacelle D3D12 runtime: ${CRUCIBLE_NACELLE_LIB}")
  endif()
endif()

if(APPLE OR WIN32 OR (CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB) OR CRUCIBLE_NACELLE_LIB)
  add_subdirectory(tools/crucible-run)
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
