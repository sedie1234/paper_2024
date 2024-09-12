#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Refine/RefineOps/Split/ONNXRefineSplit.inc"
}

namespace onnx_mlir {
namespace refine{

#include "build/src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

LogicalResult RefineSplitOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<RefineSplitOp>(op);
    RefineSplitOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void RefineSplitOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult RefineSplitOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::refine::RefineSplitOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}


} // refine
} // onnx_mlir