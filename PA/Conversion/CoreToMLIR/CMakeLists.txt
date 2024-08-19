add_onnx_mlir_rewriter(CoreToMLIR)

add_onnx_mlir_library(OMCoreToMLIR
    CoreToMLIR.cpp

    DEPENDS
    OMONNXCoreToMLIRIncGen
    MLIRArithToLLVM
    MLIRTransforms

    LINK_LIBS PUBLIC
    OMONNXOps
    OMONNXToKrnl
    OMRefineOps
    OMCoreOps
    MLIRArithToLLVM
    PARuntime

    ACCEL_INCLUDE_DIRS PRIVATE
    ${PA_INCLUDE_PATH}
)
