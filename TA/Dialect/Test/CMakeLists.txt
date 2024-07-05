add_subdirectory(TestOps)

add_onnx_mlir_dialect(Test test)
add_onnx_mlir_dialect_doc(test Test.td)

add_onnx_mlir_library(OMTestOps
  TestOps.cpp
  TestOps/Add/Add.cpp

  DEPENDS
  # OMLayoutHelper
  OMTestIncGen
  OMONNXTestAddIncGen
  OMShapeInferenceOpInterfaceIncGen
  

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  # OMLayoutHelper
  OMShapeHelperOpInterface
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
