add_subdirectory(CoreOps)

add_onnx_mlir_dialect(Core core)
add_onnx_mlir_dialect_doc(core Core.td)

add_onnx_mlir_library(OMCoreOps
  CoreOps.cpp
  CoreOps/Alloc/Alloc.cpp
  CoreOps/Read/Read.cpp
  CoreOps/Start/Start.cpp
  CoreOps/Wait/Wait.cpp
  CoreOps/Write/Write.cpp

  DEPENDS
  # OMLayoutHelper
  OMCoreIncGen
  OMONNXCoreAllocIncGen
  OMONNXCoreReadIncGen
  OMONNXCoreStartIncGen
  OMONNXCoreWaitIncGen
  OMONNXCoreWriteIncGen
  OMShapeInferenceOpInterfaceIncGen
  
  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps 
  OMShapeHelperOpInterface
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${PA_INCLUDE_PATH}
  )
