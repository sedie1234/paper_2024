#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Refine/RefineOps/Add/ONNXRefineAdd.inc"
}

namespace onnx_mlir {
namespace refine{

#include "build/src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

LogicalResult RefineAddOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<RefineAddOp>(op);
    RefineAddOp::Adaptor operandAdaptor(operands);

    DimsExpr outputDims;

    setOutputDims(outputDims);

    return success();

}

void RefineAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult RefineAddOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::refine::RefineAddOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}


} // refine
} // onnx_mlir
