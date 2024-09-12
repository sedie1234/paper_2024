#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Refine/RefineOps/Resize/ONNXRefineResize.inc"
}

namespace onnx_mlir {
namespace refine{

#include "build/src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

LogicalResult RefineResizeOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<RefineResizeOp>(op);
    RefineResizeOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void RefineResizeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult RefineResizeOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::refine::RefineResizeOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}


} // refine
} // onnx_mlir
