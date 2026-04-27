add_llvm_library(TIMVXPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        mlir-timvx-plugin.cpp

        DEPENDS
        MLIRTIMVX
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRTIMVX
        )
target_include_directories(
        TIMVXPlugin
        PRIVATE
        "${MLIR_TIMVX_BINARY_DIR}/include"
)
