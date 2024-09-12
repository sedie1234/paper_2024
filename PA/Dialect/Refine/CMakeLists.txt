add_subdirectory(RefineOps)

add_onnx_mlir_dialect(Refine refine)
add_onnx_mlir_dialect_doc(refine Refine.td)

add_onnx_mlir_library(OMRefineOps
  RefineOps.cpp
  RefineOps/Conv/Conv.cpp
  RefineOps/Sigmoid/Sigmoid.cpp
  RefineOps/Mul/Mul.cpp
  RefineOps/Split/Split.cpp
  RefineOps/Add/Add.cpp
  RefineOps/Concat/Concat.cpp
  RefineOps/Maxpool/Maxpool.cpp
  RefineOps/Resize/Resize.cpp
  RefineOps/Transpose/Transpose.cpp
  RefineOps/Softmax/Softmax.cpp
  RefineOps/Slice/Slice.cpp
  RefineOps/Sub/Sub.cpp
  RefineOps/Div/Div.cpp

  DEPENDS
  # OMLayoutHelper
  OMRefineIncGen
  OMONNXRefineConvIncGen
  OMONNXRefineSigmoidIncGen
  OMONNXRefineMulIncGen
  OMONNXRefineSplitIncGen
  OMONNXRefineAddIncGen
  OMONNXRefineConcatIncGen
  OMONNXRefineMaxpoolIncGen
  OMONNXRefineResizeIncGen
  OMONNXRefineTransposeIncGen
  OMONNXRefineSoftmaxIncGen
  OMONNXRefineSliceIncGen
  OMONNXRefineSubIncGen
  OMONNXRefineDivIncGen
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
