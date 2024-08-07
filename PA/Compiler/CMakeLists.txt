get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_library(OMPACompilerOptions
    PACompilerOptions.cpp

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${PA_SRC_ROOT}
    ${PA_BIN_ROOT}
    ${PA_ONNX_MLIR_SRC_ROOT}
    ${PA_ONNX_MLIR_BIN_ROOT}

    LINK_LIBS PUBLIC
    ${OMLibs}
    OMCompilerOptions

    ACCEL_INCLUDE_DIRS PRIVATE
    ${PA_ONNX_MLIR_SRC_ROOT}
    ${PA_ONNX_MLIR_BIN_ROOT}
)

add_onnx_mlir_library(OMPACompilerUtils
    PACompilerUtils.cpp

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${PA_SRC_ROOT}
    ${PA_BIN_ROOT}
    ${PA_ONNX_MLIR_SRC_ROOT}
    ${PA_ONNX_MLIR_BIN_ROOT}

    LINK_LIBS PUBLIC
    ${OMLibs}
    OMPACompilerOptions
    OMCompilerPasses

    ACCEL_INCLUDE_DIRS PRIVATE
    ${PA_ONNX_MLIR_SRC_ROOT}
    ${PA_ONNX_MLIR_BIN_ROOT}
)