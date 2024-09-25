add_onnx_mlir_rewriter(CoreOpt)

add_onnx_mlir_library(OMCoreOpt
    CoreOpt.cpp

    DEPENDS
    OMONNXCoreOptIncGen
    MLIRArithToLLVM
    MLIRTransforms

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
