#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

// #include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
// #include "src/Accelerators/TA/Dialect/Test/TestOps/OpHelper.hpp"
// #include "src/Accelerators/TA/Support/LayoutHelper.hpp"
#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

namespace onnx_mlir{
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



