#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
#include "src/Accelerators/TA/Pass/TAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir{


bool isUniBroadcatableFirstToSecond(Value A, Value B) {
  if (!hasStaticShape(A.getType()) || !hasStaticShape(B.getType()))
    return false;
  ArrayRef<int64_t> aDims = getShape(A.getType());
  ArrayRef<int64_t> bDims = getShape(B.getType());
  // A and B have exactly the same static shape.
  if (aDims == bDims)
    return false;
  // aDims size > bDims size: not unidirectional broadcasting from A to B, but B
  // to A.
  if (aDims.size() > bDims.size())
    return false;
  // Pre-pad A's shape with dims 1 so that two shapes have the same size.
  SmallVector<int64_t> paddedADims(bDims.size(), 1);
  for (unsigned i = 0; i < aDims.size(); ++i)
    paddedADims[i + bDims.size() - aDims.size()] = aDims[i];
  // Check unidirectional broadcasting.
  return llvm::all_of(llvm::zip(paddedADims, bDims), [](auto v) {
    return ((std::get<0>(v) == 1 && std::get<1>(v) != 1) ||
            (std::get<0>(v) == std::get<1>(v)));
  });
}

bool isDefinedByONNXConstantOp(Value v) {
  return isa_and_present<ONNXConstantOp>(v.getDefiningOp());
}

struct RewriteONNXForTestPass
    : public PassWrapper<RewriteONNXForTestPass, OperationPass<ModuleOp>> {

    StringRef getArgument() const override { return "rewrite-onnx-for-test"; }

    StringRef getDescription() const override { 
        return "Rewirte ONNX ops for Test";
    }

    RewriteONNXForTestPass() = default;
    RewriteONNXForTestPass(mlir::ArrayRef<std::string> execNodesOnCpu)
        : execNodesOnCpu(execNodesOnCpu) {}
    void runOnOperation() final;

public:
    mlir::ArrayRef<std::string> execNodesOnCpu = mlir::ArrayRef<std::string>();
};



#include "src/Accelerators/TA/Conversion/ONNXToTest/ONNXRewriteONNXForTest.inc"


void RewriteONNXForTestPass::runOnOperation() {
    ModuleOp module = getOperation();

    onnx_mlir::DimAnalysis dimAnalysis(module);
    dimAnalysis.analyze();

    ConversionTarget target(getContext());

    target.addLegalDialect<ONNXDialect, test::TestDialect, func::FuncDialect>();

    // addDynamicallyLegalOpFor<ONNXBatchNoramlizationInferenceModeOp>(
    //     &target, &dimAnalysis, execNodesOnCpu
    // );

    target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
        return !((isDefinedByONNXConstantOp(op.getA()) &&
                    isUniBroadcatableFirstToSecond(op.getA(), op.getB())) ||
                 (isDefinedByONNXConstantOp(op.getB()) && 
                    isUniBroadcatableFirstToSecond(op.getB(), op.getA())));
    });

    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    
    if(failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> createRewriteONNXForTestPass() {
    return std::make_unique<RewriteONNXForTestPass>();
}

std::unique_ptr<mlir::Pass> createRewriteONNXForTestPass(
        mlir::ArrayRef<std::string> execNodesOnCpu) {
    
    return std::make_unique<RewriteONNXForTestPass>(execNodesOnCpu);
}

} // namespace onnx_mlir
