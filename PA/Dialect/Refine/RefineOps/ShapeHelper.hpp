#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

// #include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
// #include "src/Accelerators/TA/Dialect/Test/TestOps/OpHelper.hpp"
// #include "src/Accelerators/TA/Support/LayoutHelper.hpp"
#include "src/Accelerators/PA/Dialect/Refine/RefineOps.hpp"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

namespace onnx_mlir{
namespace refine{

#define DECLARE_SHAPE_HELPER_REFINE(SHAPE_HELPER)                         \
    class SHAPE_HELPER : public ONNXOpShapeHelper {                     \
    public:                                                             \
        SHAPE_HELPER(mlir::Operation *op,                               \
            mlir::ArrayRef<mlir::Value> operands = {},                  \
            IndexExprBuilder *ieBuilder = nullptr,                      \
            IndexExprScope *scope = nullptr)                            \
            : ONNXOpShapeHelper(op, operands, ieBuilder, scope) {}      \
        virtual ~SHAPE_HELPER() {}                                      \
        mlir::LogicalResult computeShape() final;                       \
    };
DECLARE_SHAPE_HELPER_REFINE(RefineConvOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineSigmoidOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineMulOpShapeHelper)

#undef DECLARE_SHAPE_HELPER_TEST

}

}



