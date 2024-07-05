# SPDX-License-Identifier: Apache-2.0

# When an accelerator is specified to be built in ONNX_MLIR_ACCELERATORS,
# if for whatever reason we do not want to build it on a specific arch,
# XX_ENABLED variable can be used to control that. For example, if we
# only want to compile NNPA on s390x, we can do:
#
# if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
#   set(NNPA_ENABLED 1 BOOL PARENT_SCOPE)
#   ...
# endif()

set(TA_ENABLED 1 BOOL PARENT_SCOPE)

set(TA_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TA_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(TA_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(TA_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(TA_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set(TA_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(TA_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

add_subdirectory(Dialect)
add_subdirectory(Conversion)
# add_subdirectory(Pass)
add_subdirectory(Compiler)


add_onnx_mlir_library(OMTAAccel
  TAAccelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${TA_ONNX_MLIR_SRC_ROOT}
  ${TA_SRC_ROOT}
  ${TA_BIN_ROOT}
  ${TA_INCLUDE_PATH}

  LINK_LIBS PUBLIC
  onnx
  OMTACompilerUtils
  OMAccelerator
  OMTestOps

  
  )
