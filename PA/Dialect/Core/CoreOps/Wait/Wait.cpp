#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Core/CoreOps/Wait/ONNXCoreWait.inc"
}

namespace onnx_mlir {
namespace core{

#include "build/src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

LogicalResult CoreWaitOpShapeHelper::computeShape(){
    auto waitOp = llvm::dyn_cast<CoreWaitOp>(op);
    CoreWaitOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void CoreWaitOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;
}

LogicalResult CoreWaitOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::core::CoreWaitOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

} // core
} // onnx_mlir