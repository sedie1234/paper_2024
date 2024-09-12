add_onnx_mlir_rewriter(RefineToCore)

add_onnx_mlir_library(OMRefineToCore
    RefineToCore.cpp

    DEPENDS
    OMONNXRefineToCoreIncGen
    MLIRArithToLLVM
    MLIRTransforms
    OMONNXOps

    LINK_LIBS PUBLIC
    OMONNXOps
    OMONNXToKrnl
    OMCoreOps
    OMRefineOps
    MLIRArithToLLVM
    PARuntime

    ACCEL_INCLUDE_DIRS PRIVATE
    ${PA_INCLUDE_PATH}
)
