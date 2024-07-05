// #include "src/Accelerators/TA/Dialect/Test/TestOps/ShapeHelper.hpp"

#include "src/Accelerators/TA/Dialect/Test/TestOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
#include "src/Accelerators/TA/Dialect/Test/TestOps/Add/ONNXTestAdd.inc"
}

// #ifndef GET_OP_CLASSES
// #define GET_OP_CLASSES

// #endif

namespace onnx_mlir {
namespace test{

#include "build/src/Accelerators/TA/Dialect/Test/TestOps.hpp.inc"

void TestAddOp::build(
    OpBuilder &builder, OperationState &state, Value lhs, Value rhs){

    state.addOperands({lhs, rhs});

    // if(lhs.getType() != rhs.getType()){
    //     emitError(builder.getUnknownLoc(), "type error");
    //     return;
    // }
    state.addTypes(lhs.getType());
}

LogicalResult TestAddOpShapeHelper::computeShape(){
    auto addOp = llvm::dyn_cast<TestAddOp>(op);
    TestAddOp::Adaptor operandAdaptor(operands);
    // Value input = operandAdaptor.getIn();

    DimsExpr outputDims;

    // SmallVector<IndexExpr, 4> inputDims;
    // createIE->getShapeAsDims(input, inputDims);
    // int64_t rank = inputDims.size();

    setOutputDims(outputDims);

    return success();

}

void TestAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context){

    // results.insert<ReplaceONNXAddPattern>(context);
    return;

}

LogicalResult TestAddOp::inferShapes(
       std::function<void(mlir::Region &)> doShapeInference) {
    // Value input = getIn();
    // if (isa<NoneType>(input.getType()) || !hasRankedType(input))
    //     return success();
    
    return success();
}

onnx_mlir::ONNXOpShapeHelper *onnx_mlir::test::TestAddOp::getShapeHelper(
        mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
    return nullptr;
}

mlir::Attribute onnx_mlir::test::TestDialect::parseAttribute(
        mlir::DialectAsmParser &parser, mlir::Type typel) const {

    return {};
}

void onnx_mlir::test::TestDialect::printAttribute(
        mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
    return;
}


}
}



