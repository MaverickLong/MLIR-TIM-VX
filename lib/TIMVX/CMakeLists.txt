add_mlir_dialect_library(MLIRTIMVX
        TIMVXTypes.cpp
        TIMVXDialect.cpp
        TIMVXOps.cpp
        TIMVXPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TIMVX

        DEPENDS
        MLIRTIMVXOpsIncGen
        MLIRTIMVXEnumsIncGen
        MLIRTIMVXPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIREmitCDialect
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRQuantDialect
        MLIRTosaDialect
        MLIRTransformUtils
        )
