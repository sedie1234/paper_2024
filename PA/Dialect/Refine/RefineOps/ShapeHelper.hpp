#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp"
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
DECLARE_SHAPE_HELPER_REFINE(RefineSplitOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineAddOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineConcatOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineMaxpoolOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineResizeOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineTransposeOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineSoftmaxOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineSliceOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineSubOpShapeHelper)
DECLARE_SHAPE_HELPER_REFINE(RefineDivOpShapeHelper)

#undef DECLARE_SHAPE_HELPER_TEST

}

namespace core{

#define DECLARE_SHAPE_HELPER_CORE(SHAPE_HELPER)                         \
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
DECLARE_SHAPE_HELPER_CORE(CoreAllocOpShapeHelper)
DECLARE_SHAPE_HELPER_CORE(CoreReadOpShapeHelper)
DECLARE_SHAPE_HELPER_CORE(CoreStartOpShapeHelper)
DECLARE_SHAPE_HELPER_CORE(CoreWaitOpShapeHelper)
DECLARE_SHAPE_HELPER_CORE(CoreWriteOpShapeHelper)

#undef DECLARE_SHAPE_HELPER_TEST

}

}

