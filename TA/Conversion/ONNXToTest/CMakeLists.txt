add_onnx_mlir_rewriter(ONNXToTest)

add_onnx_mlir_library(OMONNXToTest
    ONNXToTest.cpp

    DEPENDS
    OMONNXONNXToTestIncGen

    LINK_LIBS PUBLIC
    OMONNXOps
    OMONNXToKrnl
    OMTestOps

    ACCEL_INCLUDE_DIRS PRIVATE
    ${TA_INCLUDE_PATH}
)

add_onnx_mlir_rewriter(RewriteONNXForTest)

add_onnx_mlir_library(OMRewriteONNXForTest
    RewriteONNXForTest.cpp

    DEPENDS
    OMONNXRewriteONNXForTestIncGen

    LINK_LIBS PUBLIC
    OMONNXOps
    OMONNXToKrnl
    OMTestOps

    ACCEL_INCLUDE_DIRS PRIVATE
    ${TA_INCLUDE_PATH}
)

