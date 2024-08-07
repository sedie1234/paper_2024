set(PA_ENABLED 1 BOOL PARENT_SCOPE)

set(PA_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PA_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(PA_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(PA_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(PA_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set(PA_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(PA_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

add_subdirectory(Dialect)
add_subdirectory(Conversion)
# add_subdirectory(Pass)
add_subdirectory(Compiler)
# add_subdirectory(Runtime)


add_onnx_mlir_library(OMPAAccel
  PAAccelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${PA_ONNX_MLIR_SRC_ROOT}
  ${PA_SRC_ROOT}
  ${PA_BIN_ROOT}
  ${PA_INCLUDE_PATH}

  LINK_LIBS PUBLIC
  onnx
  OMPACompilerUtils
  OMAccelerator
  OMRefineOps
  
  
)