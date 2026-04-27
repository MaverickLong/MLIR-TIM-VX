set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRTIMVX
  )
add_llvm_executable(mlir-timvx-opt mlir-timvx-opt.cpp)

llvm_update_compile_flags(mlir-timvx-opt)
target_link_libraries(mlir-timvx-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-timvx-opt)
