add_onnx_mlir_library(PARuntime STATIC
    PARuntime.c

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${PA_INCLUDE_PATH}
    
)

set_target_properties(PARuntime
    PROPERTIES
    LANGUAGE C_EXTENSIONS
    POSITION_INDEPENDENT_CODE TRUE
)
