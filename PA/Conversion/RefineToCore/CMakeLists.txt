add_onnx_mlir_rewriter(RefineToCore)

add_onnx_mlir_library(OMRefineToCore
    RefineToCore.cpp

    DEPENDS
    OMONNXRefineToCoreIncGen
    MLIRArithToLLVM
    MLIRTransforms
    OMONNXCoreAllocIncGen
    OMONNXCoreReadIncGen
    OMONNXCoreStartIncGen
    OMONNXCoreWaitIncGen
    OMONNXCoreWriteIncGen

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
