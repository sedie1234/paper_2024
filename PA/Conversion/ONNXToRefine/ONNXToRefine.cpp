#include "src/Accelerators/PA/Dialect/Refine/RefineOps.hpp"
#include "src/Accelerators/PA/Pass/PAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm-c/Core.h"

using namespace mlir;
namespace onnx_mlir {
    
namespace {

#include "src/Accelerators/PA/Conversion/ONNXToRefine/ONNXONNXToRefine.inc"


struct ONNXToRefineLoweringPass
    : public PassWrapper<ONNXToRefineLoweringPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToRefineLoweringPass)

    StringRef getArgument() const override {return "convert-onnx-to-refine";}
    StringRef getDescription() const override {
        return "Lower ONNX ops to Refine ops";
    }

    ONNXToRefineLoweringPass() = default;
    ONNXToRefineLoweringPass(const ONNXToRefineLoweringPass &pass)
        : PassWrapper<ONNXToRefineLoweringPass, OperationPass<ModuleOp>>() {}
    ONNXToRefineLoweringPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

    // void getDependentDialects(DialectRegistry &registry) const override {
    //     registry.insert<bufferization::BufferizationDialect>();
    // }

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnSCpu",
        llvm::cl::desc("ONNX To Refine Lowering Pass"),
        llvm::cl::ZeroOrMore};

};
} // end anonymous namespace

class ConvertONNXConvToRefine : public OpRewritePattern<ONNXConvOp> {
public:
    using OpRewritePattern<ONNXConvOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXConvOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        auto auto_padAttr = op->getAttr("auto_pad");
        auto dilationsAttr = op->getAttr("dilations");
        auto groupAttr = op->getAttr("group");
        auto kernel_shapeAttr = op->getAttr("kernel_shape");
        auto padsAttr = op->getAttr("pads");
        auto stridesAttr = op->getAttr("strides");
        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        Value bias = op.getOperand(2);

        if(bias.getType().isa<NoneType>()){
            auto kernelType = kernel.getType().cast<RankedTensorType>();
            int64_t kernelDim0 = kernelType.getShape()[0];

            // kernel의 0번 랭크의 크기를 shape로 가지는 tensor를 생성 (모두 0으로 초기화)
            auto elementType = rewriter.getF32Type(); // 텐서의 원소 타입 (float)
            auto tensorType = RankedTensorType::get({kernelDim0}, elementType);
            auto tensorAttr = DenseElementsAttr::get(tensorType, rewriter.getZeroAttr(elementType));

            bias = rewriter.create<mlir::arith::ConstantOp>(loc, tensorType, tensorAttr);
        }

        auto refineConvOp = rewriter.create<refine::RefineConvOp>(
            loc,
            resultType,
            input,
            kernel,
            bias,
            kernel_shapeAttr.cast<ArrayAttr>(),
            padsAttr.cast<ArrayAttr>(),
            stridesAttr.cast<ArrayAttr>(),
            onnx_node_name.cast<StringAttr>()
        );

        rewriter.replaceOp(op, refineConvOp);

        return success();
    }
};

class ConvertONNXSigmoidToRefine : public OpRewritePattern<ONNXSigmoidOp> {
public:
    using OpRewritePattern<ONNXSigmoidOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXSigmoidOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand();
        auto onnx_node_name = op->getAttr("onnx_node_name");

        auto refineSigmoidOp = rewriter.create<refine::RefineSigmoidOp>(loc, resultType, input, onnx_node_name.cast<StringAttr>());

        rewriter.replaceOp(op, refineSigmoidOp);

        return success();
    }
};

class ConvertONNXMulToRefine : public OpRewritePattern<ONNXMulOp> {
public:
    using OpRewritePattern<ONNXMulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXMulOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();
        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value X = op.getOperand(0);
        Value Y = op.getOperand(1);

        auto refineMulOp = rewriter.create<refine::RefineMulOp>(loc, resultType, X, Y, onnx_node_name.cast<StringAttr>());

        rewriter.replaceOp(op, refineMulOp);

        return success();
    }
};

void ONNXToRefineLoweringPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertONNXConvToRefine, ConvertONNXSigmoidToRefine, 
        ConvertONNXMulToRefine>(&getContext());

    // if(failed(applyPartialConversion(module, target, std::move(patterns))))
    //     signalPassFailure();

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

std::unique_ptr<Pass> createONNXToRefinePass() {
    return std::make_unique<ONNXToRefineLoweringPass>();
}

std::unique_ptr<Pass> createONNXToRefinePass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<ONNXToRefineLoweringPass>(execNodesOnCpu);
}


} // namespace onnx_mlir








