#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Refine/RefineOps/Mul/ONNXRefineMul.inc"
}

namespace onnx_mlir {
namespace refine{

#include "build/src/Accelerators/PA/Dialect/Refine/RefineOps.hpp.inc"

// void RefineMulOp::build(
//     OpBuilder &builder, OperationState &state, Value input, Value kernel, Value bias,
//     ArrayAttr kernel_shape, ArrayAttr padding, ArrayAttr strides){

//     state.addOperands({input, kernel, bias, kernel_shape, padding, strides});
// //overide
// }

LogicalResult RefineMulOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<RefineMulOp>(op);
    RefineMulOp::Adaptor operandAdaptor(operands);
    // Value input = operandAdaptor.getIn();

    DimsExpr outputDims;

    // SmallVector<IndexExpr, 4> inputDims;
    // createIE->getShapeAsDims(input, inputDims);
    // int64_t rank = inputDims.size();

    setOutputDims(outputDims);

    return success();

}

void RefineMulOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult RefineMulOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::refine::RefineMulOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

// mlir::Attribute onnx_mlir::refine::RefineDialect::parseAttribute(
//         mlir::DialectAsmParser &parser, mlir::Type typel) const {

//     return {};
// }

// void onnx_mlir::refine::RefineDialect::printAttribute(
//         mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
//     return;
// }

} // refine
} // onnx_mlir