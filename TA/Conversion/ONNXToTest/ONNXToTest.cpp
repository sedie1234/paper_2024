#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
#include "src/Accelerators/TA/Pass/TAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

#include "src/Accelerators/TA/Conversion/ONNXToTest/ONNXONNXToTest.inc"

struct ONNXToTestLoweringPass
    : public PassWrapper<ONNXToTestLoweringPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToTestLoweringPass)

    StringRef getArgument() const override { return "convert-onnx-to-test"; }

    StringRef getDescription() const override {
        return "Lower ONNX ops to Test ops";
    }

    ONNXToTestLoweringPass() = default;
    ONNXToTestLoweringPass(const ONNXToTestLoweringPass &pass)
        : PassWrapper<ONNXToTestLoweringPass, OperationPass<ModuleOp>>() {}
    ONNXToTestLoweringPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnCpu",
        llvm::cl::desc("ONNX To Test Lowering Pass Test"),
        llvm::cl::ZeroOrMore
    };

};
} // end anonymous namespace

void ONNXToTestLoweringPass::runOnOperation() {
    ModuleOp module = getOperation();

    onnx_mlir::DimAnalysis dimAnalysis(module);
    dimAnalysis.analyze();

    ConversionTarget target(getContext());

    target.addLegalDialect<ONNXDialect, test::TestDialect, KrnlDialect,
        func::FuncDialect, arith::ArithDialect>();

    RewritePatternSet combinedPatterns(&getContext());
    combinedPatterns.insert<ReplaceONNXAddPattern>(&getContext());

    (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

    //addDynamicallyLegalOpFor<ONNXAddOp>(&target, &dimAnalysis, execNodesOnCpu);

}

std::unique_ptr<Pass> createONNXToTestPass() {
    return std::make_unique<ONNXToTestLoweringPass>();
}

std::unique_ptr<Pass> createONNXToTestPass(
        mlir::ArrayRef<std::string> execNodesOnCpu) {
    return std::make_unique<ONNXToTestLoweringPass>(execNodesOnCpu);
}

} // namespace onnx_mlir

