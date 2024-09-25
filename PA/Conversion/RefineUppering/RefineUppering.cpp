#include "src/Accelerators/PA/Dialect/Refine/RefineOps.hpp"
#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp"
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

#include "src/Accelerators/PA/Conversion/RefineUppering/ONNXRefineUppering.inc"

struct RefineUpperingPass
    : public PassWrapper<RefineUpperingPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineUpperingPass)

    StringRef getArgument() const override {return "refine uppering";}
    StringRef getDescription() const override {
        return "Upper Refine Ops";
    }

    RefineUpperingPass() = default;
    RefineUpperingPass(const RefineUpperingPass &pass)
        : PassWrapper<RefineUpperingPass, OperationPass<ModuleOp>>() {}
    RefineUpperingPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnSCpu",
        llvm::cl::desc("Refine Uppering"),
        llvm::cl::ZeroOrMore};
};
} // end anonymous namespace

class ConvertRefineUppering : public OpRewritePattern<refine::RefineConvOp> {
public:
    using OpRewritePattern<refine::RefineConvOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineConvOp op, PatternRewriter &rewriter) const override{

        // if set fused attr, failure.
        if(op->hasAttr("fused")){
            return failure();
        }

        int conv_use_count = 0;
        int sigmoid_use_count = 0;

        auto ID = op->getAttr("ID");
        auto castID = ID.cast<StringAttr>();

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        Value bias;

        if (op.getNumOperands() > 2) {
            bias = op.getOperand(2);  // If bias exists, use it
        } else {
            bias = mlir::Value();     // If bias does not exist, set bias as NoneType (empty Value)
        }

        auto kernel_shapeAttr = op->getAttr("kernel_shape");
        auto padsAttr = op->getAttr("padding");
        auto stridesAttr = op->getAttr("strides");
        auto onnx_node_name = op->getAttr("ID");
        

        Operation* sigmoidOp;
        Operation* mulOp;

        for(auto it : op->getUsers()){
            // llvm::outs() << it->getName() << "\n";
            conv_use_count++;
            if(it->getName().getStringRef() == "refine.Sigmoid"){
                sigmoidOp = dyn_cast<refine::RefineSigmoidOp>(it);
            }else if(it->getName().getStringRef() == "refine.Mul"){
                mulOp = dyn_cast<refine::RefineMulOp>(it);
            }else{
                return failure();
            }
        }

        if(conv_use_count != 2)
            return failure();

        for(auto it : sigmoidOp->getUsers()){
            sigmoid_use_count++;
            if(it == mulOp){
                
            }else{
                return failure();
            }
        }

        if((conv_use_count != 2) || (sigmoid_use_count != 1))
            return failure();


        // (*userOp)->dump();
        // userOp++;
        // (*userOp)->dump();

        // fused attr set for just one path
        op->setAttr("fused", rewriter.getUnitAttr());

        auto loc = op.getLoc();
        Type resultType = mulOp->getResult(0).getType();

        if(!bias.getType().isa<NoneType>()){
            auto refineFusedConvOp = rewriter.create<refine::RefineFusedConvOp>(
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

            rewriter.replaceOp(mulOp, refineFusedConvOp);
            // rewriter.eraseOp(sigmoidOp);
            // rewriter.eraseOp(mulOp);
        }else{

            auto NoneValue = Value();

            auto refineFusedConvOp = rewriter.create<refine::RefineFusedConvOp>(
                loc,
                resultType,
                input,
                kernel,
                NoneValue,
                kernel_shapeAttr.cast<ArrayAttr>(),
                padsAttr.cast<ArrayAttr>(),
                stridesAttr.cast<ArrayAttr>(),
                onnx_node_name.cast<StringAttr>()
            );

            // rewriter.replaceOp(op, refineFusedConvOp);
            // rewriter.replaceOp(sigmoidOp, refineFusedConvOp);
            // rewriter.replaceOp(mulOp, refineFusedConvOp);
            rewriter.replaceOp(mulOp, refineFusedConvOp.getResult());
            // rewriter.eraseOp(sigmoidOp);
            // rewriter.eraseOp(mulOp);

        }


        return success();
    }
};


void RefineUpperingPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertRefineUppering>(&getContext());

    OpBuilder builder(module.getContext());

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

std::unique_ptr<Pass> createRefineUpperingPass() {
    return std::make_unique<RefineUpperingPass>();
}

std::unique_ptr<Pass> createRefineUpperingPass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<RefineUpperingPass>(execNodesOnCpu);
}


} // namespace onnx_mlir








