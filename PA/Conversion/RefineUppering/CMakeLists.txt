add_onnx_mlir_rewriter(RefineUppering)

add_onnx_mlir_library(OMRefineUppering
    RefineUppering.cpp

    DEPENDS
    OMONNXRefineUpperingIncGen
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
