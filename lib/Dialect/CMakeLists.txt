set(CRUCIBLE_DIALECT_INC_DIR "${CMAKE_BINARY_DIR}/include/crucible/Dialect")
file(MAKE_DIRECTORY "${CRUCIBLE_DIALECT_INC_DIR}")

set(CRUCIBLE_TD_INCLUDE_FLAGS
  "-I${CMAKE_SOURCE_DIR}/include"
)
foreach(inc_dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND CRUCIBLE_TD_INCLUDE_FLAGS "-I${inc_dir}")
endforeach()

set(CRUCIBLE_TD_DEPS
  "${CMAKE_SOURCE_DIR}/include/crucible/Dialect/CrucibleDialect.td"
  "${CMAKE_SOURCE_DIR}/include/crucible/Dialect/CrucibleTypes.td"
  "${CMAKE_SOURCE_DIR}/include/crucible/Dialect/CrucibleEnums.td"
  "${CMAKE_SOURCE_DIR}/include/crucible/Dialect/CrucibleOps.td"
)

function(crucible_mlir_tablegen output_file input_td)
  add_custom_command(
    OUTPUT "${CRUCIBLE_DIALECT_INC_DIR}/${output_file}"
    COMMAND "${CRUCIBLE_MLIR_TBLGEN}"
      ${ARGN}
      ${CRUCIBLE_TD_INCLUDE_FLAGS}
      "${CMAKE_SOURCE_DIR}/include/crucible/Dialect/${input_td}"
      -o "${CRUCIBLE_DIALECT_INC_DIR}/${output_file}"
    DEPENDS ${CRUCIBLE_TD_DEPS}
    COMMENT "Generating ${output_file}"
    VERBATIM
  )
endfunction()

crucible_mlir_tablegen(CrucibleDialect.h.inc   CrucibleDialect.td -gen-dialect-decls -dialect=crucible)
crucible_mlir_tablegen(CrucibleDialect.cpp.inc CrucibleDialect.td -gen-dialect-defs  -dialect=crucible)
crucible_mlir_tablegen(CrucibleEnums.h.inc     CrucibleEnums.td   -gen-enum-decls)
crucible_mlir_tablegen(CrucibleEnums.cpp.inc   CrucibleEnums.td   -gen-enum-defs)
crucible_mlir_tablegen(CrucibleEnumAttrs.h.inc   CrucibleEnums.td -gen-attrdef-decls -attrdefs-dialect=crucible)
crucible_mlir_tablegen(CrucibleEnumAttrs.cpp.inc CrucibleEnums.td -gen-attrdef-defs  -attrdefs-dialect=crucible)
crucible_mlir_tablegen(CrucibleTypes.h.inc     CrucibleTypes.td   -gen-typedef-decls -typedefs-dialect=crucible)
crucible_mlir_tablegen(CrucibleTypes.cpp.inc   CrucibleTypes.td   -gen-typedef-defs  -typedefs-dialect=crucible)
crucible_mlir_tablegen(CrucibleOps.h.inc       CrucibleOps.td     -gen-op-decls)
crucible_mlir_tablegen(CrucibleOps.cpp.inc     CrucibleOps.td     -gen-op-defs)

set(CRUCIBLE_DIALECT_GENERATED_HEADERS
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleDialect.h.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleDialect.cpp.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleEnums.h.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleEnums.cpp.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleEnumAttrs.h.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleEnumAttrs.cpp.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleTypes.h.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleTypes.cpp.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleOps.h.inc"
  "${CRUCIBLE_DIALECT_INC_DIR}/CrucibleOps.cpp.inc"
)

add_custom_target(CrucibleDialectIncGen DEPENDS ${CRUCIBLE_DIALECT_GENERATED_HEADERS})

add_library(crucible_dialect STATIC
  CrucibleDialect.cpp
)
add_dependencies(crucible_dialect CrucibleDialectIncGen)
target_include_directories(crucible_dialect PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)
target_compile_options(crucible_dialect PRIVATE ${CRUCIBLE_COMMON_FLAGS})
target_link_libraries(crucible_dialect PUBLIC MLIRIR MLIRSideEffectInterfaces MLIRBytecodeOpInterface)
