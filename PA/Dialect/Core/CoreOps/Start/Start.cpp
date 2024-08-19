#include "src/Accelerators/PA/Dialect/Core/CoreOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Core/CoreOps/Start/ONNXCoreStart.inc"
}

namespace onnx_mlir {
namespace core{

#include "build/src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

LogicalResult CoreStartOpShapeHelper::computeShape(){
    auto startOp = llvm::dyn_cast<CoreStartOp>(op);
    CoreStartOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();
}

void CoreStartOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;
}

LogicalResult CoreStartOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::core::CoreStartOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

} // core
} // onnx_mlir
