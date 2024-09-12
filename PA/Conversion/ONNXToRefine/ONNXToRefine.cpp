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

        // auto auto_padAttr = op->getAttr("auto_pad");
        // auto dilationsAttr = op->getAttr("dilations");
        // auto groupAttr = op->getAttr("group");
        auto kernel_shapeAttr = op->getAttr("kernel_shape");
        auto padsAttr = op->getAttr("pads");
        auto stridesAttr = op->getAttr("strides");
        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        Value bias = op.getOperand(2);

        if(!bias.getType().isa<NoneType>()){
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
        }else{
            // auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
            // int64_t kernelDim0 = kernelType.getShape()[0];

            // auto elementType = rewriter.getF32Type();
            // auto tensorType = RankedTensorType::get({kernelDim0}, elementType);

            // auto llvmF32PtrType = LLVM::LLVMPointerType::get(elementType);
            // auto int32Type = rewriter.getIntegerType(32);

            // auto size = rewriter.getIntegerAttr(int32Type, kernelDim0);
            // auto allocSize = rewriter.create<arith::ConstantOp>(loc, int32Type, size);
            // Value biasMem = rewriter.create<LLVM::AllocaOp>(loc, llvmF32PtrType, allocSize, 0);

            // for(int64_t i=0; i<kernelDim0; i++){
            //     Value index = rewriter.create<arith::ConstantIndexOp>(loc, i);
            //     Value gep = rewriter.create<LLVM::GEPOp>(loc, llvmF32PtrType, biasMem, ValueRange{index});
            //     Value value = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0f));
            //     rewriter.create<LLVM::StoreOp>(loc, value, gep);
            // }

            // auto memrefType = MemRefType::get({kernelDim0}, rewriter.getF32Type());
            // Value biasTensor = rewriter.create<UnrealizedConversionCastOp>(loc, tensorType, biasMem).getResult(0);
            auto NoneValue = Value();

            auto refineConvOp = rewriter.create<refine::RefineConvOp>(
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

            rewriter.replaceOp(op, refineConvOp);
        }


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

class ConvertONNXAddToRefine : public OpRewritePattern<ONNXAddOp> {
public:
    using OpRewritePattern<ONNXAddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXAddOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        if(op->hasAttr("onnx_node_name")){
            auto onnx_node_name = op->getAttr("onnx_node_name");

            Value X = op.getOperand(0);
            Value Y = op.getOperand(1);

            auto refineAddOp = rewriter.create<refine::RefineAddOp>(loc, resultType, X, Y, onnx_node_name.cast<StringAttr>());

            rewriter.replaceOp(op, refineAddOp);
        }else{
            //TBD
            StringAttr tempStringAttr = rewriter.getStringAttr("tempAddID");

            Value X = op.getOperand(0);
            Value Y = op.getOperand(1);

            auto refineAddOp = rewriter.create<refine::RefineAddOp>(loc, resultType, X, Y, tempStringAttr);

            rewriter.replaceOp(op, refineAddOp);

        }


        return success();
    }
};

class ConvertONNXConcatToRefine : public OpRewritePattern<ONNXConcatOp> {
public:
    using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXConcatOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        auto inputs = op.getOperands();

        auto numInputs = inputs.size();

        auto axisAttr = op->getAttr("axis").dyn_cast<mlir::IntegerAttr>();
        // int64_t axis = axisAttr.getInt();
        int64_t axis = axisAttr.getValue().getSExtValue();

        auto type0 = inputs[0].getType().cast<RankedTensorType>();
        auto type1 = inputs[1].getType().cast<RankedTensorType>();

        auto shape0 = type0.getShape();
        auto shape1 = type1.getShape();
        auto elementType = type0.getElementType();

        llvm::SmallVector<int64_t, 4> newShape(shape0.begin(), shape0.end());
        newShape[axis] = shape0[axis] + shape1[axis];

        auto newresultType = RankedTensorType::get(newShape, elementType);

        Value result = rewriter.create<refine::RefineConcatOp>(loc, newresultType, inputs[0], inputs[1]);
        
        Value concatResult = result;

        for(int i=2; i<numInputs; i++){
            auto concatTensorType = inputs[i].getType().cast<RankedTensorType>();
            auto concatTensorShape = concatTensorType.getShape();
            newShape[axis] = newShape[axis] + concatTensorShape[axis];
            auto concatResultType = RankedTensorType::get(newShape, elementType);
            concatResult = rewriter.create<refine::RefineConcatOp>(loc, concatResultType, result, inputs[i]);
            result = concatResult;
        }

        rewriter.replaceOp(op, result);

        return success();
    }
};

class ConvertONNXSplitToRefine : public OpRewritePattern<ONNXSplitOp> {
public:
    using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXSplitOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        Type resultType0 = op.getResult(0).getType();
        Type resultType1 = op.getResult(1).getType();
        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value input = op.getOperand(0);
        Value split = op.getOperand(1);
        auto axisAttr = op->getAttr("axis").dyn_cast<mlir::IntegerAttr>();
        // int64_t axis = axisAttr.getInt();
        int64_t axis = axisAttr.getValue().getSExtValue();



        auto refineSplitOp = rewriter.create<refine::RefineSplitOp>(loc, 
                                                                    TypeRange{resultType0, resultType1},
                                                                    input,
                                                                    split);

        rewriter.replaceOp(op, {refineSplitOp.getResult(0), refineSplitOp.getResult(1)});


        return success();
    }
};

class ConvertONNXMaxpoolToRefine : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
public:
    using OpRewritePattern<ONNXMaxPoolSingleOutOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        auto kernel_shapeAttr = op->getAttr("kernel_shape");
        auto padsAttr = op->getAttr("pads");
        auto stridesAttr = op->getAttr("strides");
        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value input = op.getOperand();

        auto refineMaxpoolOp = rewriter.create<refine::RefineMaxpoolOp>(
            loc,
            resultType,
            input,
            kernel_shapeAttr.cast<ArrayAttr>(),
            padsAttr.cast<ArrayAttr>(),
            stridesAttr.cast<ArrayAttr>(),
            onnx_node_name.cast<StringAttr>()
        );

        rewriter.replaceOp(op, refineMaxpoolOp);
    
        return success();
    }
};

class ConvertONNXResizeToRefine : public OpRewritePattern<ONNXResizeOp> {
public:
    using OpRewritePattern<ONNXResizeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXResizeOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value input = op.getOperand(0);

        auto refineResizeOp = rewriter.create<refine::RefineResizeOp>(loc, resultType, input);

        rewriter.replaceOp(op, refineResizeOp);
    
        return success();
    }
};

class ConvertONNXTransposeToRefine : public OpRewritePattern<ONNXTransposeOp> {
public:
    using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXTransposeOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value input = op.getOperand();
        auto permAttr = op->getAttr("perm");

        auto refineTransposeOp = rewriter.create<refine::RefineTransposeOp>(loc, resultType, input, permAttr.cast<ArrayAttr>());

        rewriter.replaceOp(op, refineTransposeOp);
    
        return success();
    }
};

class ConvertONNXSoftmaxToRefine : public OpRewritePattern<ONNXSoftmaxOp> {
public:
    using OpRewritePattern<ONNXSoftmaxOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXSoftmaxOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        auto onnx_node_name = op->getAttr("onnx_node_name");

        Value input = op.getOperand();

        auto refineSoftmaxOp = rewriter.create<refine::RefineSoftmaxOp>(
            loc,
            resultType,
            input,
            onnx_node_name.cast<StringAttr>()
        );

        rewriter.replaceOp(op, refineSoftmaxOp);
    
        return success();
    }
};

class ConvertONNXSliceToRefine : public OpRewritePattern<ONNXSliceOp> {
public:
    using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXSliceOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand(0);
        Value starts = op.getOperand(1);
        Value ends = op.getOperand(2);
        Value axes = op.getOperand(3);
        Value steps = op.getOperand(4);

        auto refineSliceOp = rewriter.create<refine::RefineSliceOp>(
            loc,
            resultType,
            input,
            starts,
            ends,
            axes,
            steps
        );

        rewriter.replaceOp(op, refineSliceOp);

    }
};

class ConvertONNXSubToRefine : public OpRewritePattern<ONNXSubOp> {
public:
    using OpRewritePattern<ONNXSubOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXSubOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        if(op->hasAttr("onnx_node_name")){
        auto onnx_node_name = op->getAttr("onnx_node_name");

            Value X = op.getOperand(0);
            Value Y = op.getOperand(1);

            auto refineSubOp = rewriter.create<refine::RefineSubOp>(loc, resultType, X, Y, onnx_node_name.cast<StringAttr>());

            rewriter.replaceOp(op, refineSubOp);
        }else{
            //TBD
        }


        return success();
    }
};

class ConvertONNXDivToRefine : public OpRewritePattern<ONNXDivOp> {
public:
    using OpRewritePattern<ONNXDivOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ONNXDivOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        if(op->hasAttr("onnx_node_name")){
            auto onnx_node_name = op->getAttr("onnx_node_name");

            Value X = op.getOperand(0);
            Value Y = op.getOperand(1);

            auto refineDivOp = rewriter.create<refine::RefineDivOp>(loc, resultType, X, Y, onnx_node_name.cast<StringAttr>());

            rewriter.replaceOp(op, refineDivOp);
        }else{
            //TBD
        }


        return success();
    }
};

void ONNXToRefineLoweringPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp, arith::ConstantOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertONNXConvToRefine, ConvertONNXSigmoidToRefine, 
        ConvertONNXMulToRefine, ConvertONNXSplitToRefine, ConvertONNXAddToRefine,
        ConvertONNXConcatToRefine, ConvertONNXMaxpoolToRefine, ConvertONNXResizeToRefine,
        ConvertONNXTransposeToRefine, ConvertONNXSoftmaxToRefine, ConvertONNXSliceToRefine,
        ConvertONNXSubToRefine, ConvertONNXDivToRefine>(&getContext());

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








