
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

namespace mlir {

//===----------------------------------------------------------------------===//
// Traits

namespace OpTrait {


/// This class provides verification for ops that are known to have the same
/// operand and result layout.
template <typename ConcreteType>
class SameOperandsAndResultLayout
    : public TraitBase<ConcreteType, SameOperandsAndResultLayout> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::success();
  }
};
} // namespace OpTrait
} // namespace mlir

#include "src/Accelerators/TA/Dialect/Test/TestDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/TA/Dialect/Test/TestAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp.inc"
