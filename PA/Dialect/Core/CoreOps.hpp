
#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include "src/Interface/ShapeInferenceOpInterface.hpp"








#include "src/Accelerators/PA/Dialect/Core/CoreDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/PA/Dialect/Core/CoreAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

