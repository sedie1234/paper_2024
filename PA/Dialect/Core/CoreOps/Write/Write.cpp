#include "src/Accelerators/PA/Dialect/Core/CoreOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Core/CoreOps/Write/ONNXCoreWrite.inc"
}

namespace onnx_mlir {
namespace core{

#include "build/src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

LogicalResult CoreWriteOpShapeHelper::computeShape(){
    auto writeOp = llvm::dyn_cast<CoreWriteOp>(op);
    CoreWriteOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void CoreWriteOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;
}

LogicalResult CoreWriteOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::core::CoreWriteOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

} // core
} // onnx_mlir