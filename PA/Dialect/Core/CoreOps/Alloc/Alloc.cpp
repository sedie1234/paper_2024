#include "src/Accelerators/PA/Dialect/Refine/RefineOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/PA/Dialect/Core/CoreOps/Alloc/ONNXCoreAlloc.inc"
}

namespace onnx_mlir {
namespace core{

#include "build/src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"

LogicalResult CoreAllocOpShapeHelper::computeShape(){
    auto allocOp = llvm::dyn_cast<CoreAllocOp>(op);
    CoreAllocOp::Adaptor operandAdaptor(operands);
    // Value input = operandAdaptor.getIn();

    DimsExpr outputDims;

    // SmallVector<IndexExpr, 4> inputDims;
    // createIE->getShapeAsDims(input, inputDims);
    // int64_t rank = inputDims.size();

    setOutputDims(outputDims);

    return success();

}

void CoreAllocOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    return;

}

LogicalResult CoreAllocOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::core::CoreAllocOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

mlir::Attribute onnx_mlir::core::CoreDialect::parseAttribute(
        mlir::DialectAsmParser &parser, mlir::Type typel) const {

    return {};
}

void onnx_mlir::core::CoreDialect::printAttribute(
        mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
    return;
}

} // core
} // onnx_mlir