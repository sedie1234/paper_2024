get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_library(OMTACompilerOptions
    TACompilerOptions.cpp

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${TA_SRC_ROOT}
    ${TA_BIN_ROOT}
    ${TA_ONNX_MLIR_SRC_ROOT}
    ${TA_ONNX_MLIR_BIN_ROOT}

    LINK_LIBS PUBLIC
    ${OMLibs}
    OMCompilerOptions

    ACCEL_INCLUDE_DIRS PRIVATE
    ${TA_ONNX_MLIR_SRC_ROOT}
    ${TA_ONNX_MLIR_BIN_ROOT}
)

add_onnx_mlir_library(OMTACompilerUtils
    TACompilerUtils.cpp

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${TA_SRC_ROOT}
    ${TA_BIN_ROOT}
    ${TA_ONNX_MLIR_SRC_ROOT}
    ${TA_ONNX_MLIR_BIN_ROOT}

    LINK_LIBS PUBLIC
    ${OMLibs}
    OMTACompilerOptions
    OMCompilerPasses

    ACCEL_INCLUDE_DIRS PRIVATE
    ${TA_ONNX_MLIR_SRC_ROOT}
    ${TA_ONNX_MLIR_BIN_ROOT}
)