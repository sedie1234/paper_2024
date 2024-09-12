#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Refine/RefineOps/Sub/ONNXRefineSub.inc"
}

namespace onnx_mlir {
namespace refine{

#include "build/src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

LogicalResult RefineSubOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<RefineSubOp>(op);
    RefineSubOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void RefineSubOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult RefineSubOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::refine::RefineSubOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}


} // refine
} // onnx_mlir
