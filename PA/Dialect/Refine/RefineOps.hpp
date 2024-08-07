
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
namespace OpTrait {

template <typename ConcreteType>
class SameOperandsAndResultLayout
    : public TraitBase<ConcreteType, SameOperandsAndResultLayout> {

public:
    static LogicalResult verifytrait(Operation *op){
        return mlir::success();
    }
};

} // namespace OpTrait
} // namespace mlir

#include "src/Accelerators/PA/Dialect/Refine/RefineDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/PA/Dialect/Refine/RefineAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

