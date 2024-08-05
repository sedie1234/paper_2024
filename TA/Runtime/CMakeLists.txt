add_onnx_mlir_library(TARuntime STATIC
    TARuntime.c

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PRIVATE
    ${TA_INCLUDE_PATH}
    
)

set_target_properties(TARuntime
    PROPERTIES
    LANGUAGE C_EXTENSIONS
    POSITION_INDEPENDENT_CODE TRUE
)