add_mlir_dialect(TIMVXOps timvx)
add_mlir_doc(TIMVXDialect TIMVXDialect TIMVX/ -gen-dialect-doc)
add_mlir_doc(TIMVXOps TIMVXOps TIMVX/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TIMVXOps.td)
mlir_tablegen(TIMVXEnums.h.inc -gen-enum-decls)
mlir_tablegen(TIMVXEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTIMVXEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS TIMVXPasses.td)
mlir_tablegen(TIMVXPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTIMVXPassesIncGen)
