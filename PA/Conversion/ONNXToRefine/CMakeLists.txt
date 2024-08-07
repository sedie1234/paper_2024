add_onnx_mlir_rewriter(ONNXToRefine)

add_onnx_mlir_library(OMONNXToRefine
    ONNXToRefine.cpp

    DEPENDS
    OMONNXONNXToRefineIncGen
    MLIRArithToLLVM
    MLIRTransforms

    LINK_LIBS PUBLIC
    OMONNXOps
    OMONNXToKrnl
    OMRefineOps
    MLIRArithToLLVM
    # PARuntime

    ACCEL_INCLUDE_DIRS PRIVATE
    ${PA_INCLUDE_PATH}
)

# add_onnx_mlir_rewriter(RewriteONNXForTest)

# add_onnx_mlir_library(OMRewriteONNXForTest
#     RewriteONNXForTest.cpp

#     DEPENDS
#     OMONNXRewriteONNXForTestIncGen

#     LINK_LIBS PUBLIC
#     OMONNXOps
#     OMONNXToKrnl
#     OMTestOps
#     TARuntime

#     ACCEL_INCLUDE_DIRS PRIVATE
#     ${TA_INCLUDE_PATH}
# )
