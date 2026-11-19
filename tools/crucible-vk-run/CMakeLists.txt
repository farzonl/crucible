add_executable(crucible-vk-run
  main.cpp
)
target_include_directories(crucible-vk-run PRIVATE ${CRUCIBLE_VULKAN_INCLUDE_DIR})
target_link_libraries(crucible-vk-run PRIVATE ${CRUCIBLE_VULKAN_LIB})
if(APPLE)
  # Ensure MoltenVK dylib rpath is embedded so crucible-vk-run runs cleanly
  # from the build tree without DYLD_LIBRARY_PATH.
  get_filename_component(_vk_lib_dir "${CRUCIBLE_VULKAN_LIB}" DIRECTORY)
  set_target_properties(crucible-vk-run PROPERTIES
    BUILD_RPATH "${_vk_lib_dir}")
endif()
