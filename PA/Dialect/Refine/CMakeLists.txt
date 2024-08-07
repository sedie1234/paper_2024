add_subdirectory(RefineOps)

add_onnx_mlir_dialect(Refine refine)
add_onnx_mlir_dialect_doc(refine Refine.td)

add_onnx_mlir_library(OMRefineOps
  RefineOps.cpp
  RefineOps/Conv/Conv.cpp
  RefineOps/Sigmoid/Sigmoid.cpp
  RefineOps/Mul/Mul.cpp

  DEPENDS
  # OMLayoutHelper
  OMRefineIncGen
  OMONNXRefineConvIncGen
  OMONNXRefineSigmoidIncGen
  OMONNXRefineMulIncGen
  OMShapeInferenceOpInterfaceIncGen
  

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  # OMLayoutHelper
  OMShapeHelperOpInterface
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${PA_INCLUDE_PATH}
  )
