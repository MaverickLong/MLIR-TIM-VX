add_mlir_dialect_library(MLIRTIMVX
        TIMVXTypes.cpp
        TIMVXDialect.cpp
        TIMVXOps.cpp
        Passes/TosaConstFold.cpp
        Passes/TosaQuantAnchor.cpp
        Passes/TosaFoldAvgPoolReduce.cpp
        Passes/TosaToTimvx.cpp
        Passes/TimvxConv1x1ToFC.cpp
        Passes/TimvxQuantResidualFuse.cpp
        Passes/TimvxToEmitC.cpp

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
        MLIRQuantUtils
        )

# Per-pass cpp files include `Common.h` from the `Passes/` subdir; expose
# this lib's source root so `#include "Passes/Common.h"` works from
# `TIMVXOps.cpp` too.
target_include_directories(obj.MLIRTIMVX PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
