set(CRUCIBLE_RUN_SOURCES
  main.cpp
  Common.cpp
)

if(APPLE)
  list(APPEND CRUCIBLE_RUN_SOURCES MetalBackend.mm)
  set_source_files_properties(MetalBackend.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")
endif()

if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  list(APPEND CRUCIBLE_RUN_SOURCES VulkanBackend.cpp)
endif()

if(WIN32 OR CRUCIBLE_NACELLE_LIB)
  list(APPEND CRUCIBLE_RUN_SOURCES D3D12Backend.cpp)
endif()

add_executable(crucible-run ${CRUCIBLE_RUN_SOURCES})
target_compile_definitions(crucible-run PRIVATE CRUCIBLE_VERSION="${PROJECT_VERSION}")
install(TARGETS crucible-run RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(APPLE)
  target_compile_definitions(crucible-run PRIVATE CRUCIBLE_RUN_HAS_METAL=1)
  target_link_libraries(crucible-run PRIVATE
    "-framework Metal"
    "-framework Foundation"
  )
endif()

if(CRUCIBLE_VULKAN_INCLUDE_DIR AND CRUCIBLE_VULKAN_LIB)
  target_compile_definitions(crucible-run PRIVATE CRUCIBLE_RUN_HAS_VULKAN=1)
  target_include_directories(crucible-run PRIVATE ${CRUCIBLE_VULKAN_INCLUDE_DIR})
  target_link_libraries(crucible-run PRIVATE ${CRUCIBLE_VULKAN_LIB})
  if(APPLE)
    # Ensure MoltenVK / vulkan-loader dylib rpath is embedded so crucible-run
    # runs cleanly from the build tree without DYLD_LIBRARY_PATH.
    get_filename_component(_vk_lib_dir "${CRUCIBLE_VULKAN_LIB}" DIRECTORY)
    set_target_properties(crucible-run PROPERTIES
      BUILD_RPATH "${_vk_lib_dir}")
  endif()
endif()

if(WIN32 OR CRUCIBLE_NACELLE_LIB)
  target_compile_definitions(crucible-run PRIVATE CRUCIBLE_RUN_HAS_D3D12=1)
  if(CRUCIBLE_NACELLE_LIB)
    target_compile_definitions(crucible-run PRIVATE
      CRUCIBLE_NACELLE_LIB_PATH="${CRUCIBLE_NACELLE_LIB}")
  endif()
  target_include_directories(crucible-run PRIVATE
    "${CMAKE_SOURCE_DIR}/third_party/DirectX-Headers/include")
  if(NOT WIN32)
    target_include_directories(crucible-run PRIVATE
      "${CMAKE_SOURCE_DIR}/third_party/DirectX-Headers/include/wsl/stubs")
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(crucible-run PRIVATE dl)
  endif()
endif()
