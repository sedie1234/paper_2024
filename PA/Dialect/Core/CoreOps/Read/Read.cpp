#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Core/CoreOps/Read/ONNXCoreRead.inc"
}

namespace onnx_mlir {
namespace core{

#include "build/src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

LogicalResult CoreReadOpShapeHelper::computeShape(){
    auto readOp = llvm::dyn_cast<CoreReadOp>(op);
    CoreReadOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void CoreReadOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;
}

LogicalResult CoreReadOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::core::CoreReadOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

} // core
} // onnx_mlir