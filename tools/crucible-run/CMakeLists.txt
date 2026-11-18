add_executable(crucible-run main.mm)
target_compile_options(crucible-run PRIVATE -fobjc-arc)
target_link_libraries(crucible-run PRIVATE
  "-framework Metal"
  "-framework Foundation"
)
