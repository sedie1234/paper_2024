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

#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp.inc"
#include "src/Accelerators/PA/Conversion/RefineToCore/ONNXRefineToCore.inc"


struct RefineToCoreLoweringPass
    : public PassWrapper<RefineToCoreLoweringPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineToCoreLoweringPass)

    StringRef getArgument() const override {return "convert-onnx-to-refine";}
    StringRef getDescription() const override {
        return "Lower Refine ops to Core ops";
    }

    RefineToCoreLoweringPass() = default;
    RefineToCoreLoweringPass(const RefineToCoreLoweringPass &pass)
        : PassWrapper<RefineToCoreLoweringPass, OperationPass<ModuleOp>>() {}
    RefineToCoreLoweringPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnSCpu",
        llvm::cl::desc("Refine To Core Lowering Pass"),
        llvm::cl::ZeroOrMore};

};
} // end anonymous namespace

class ConvertRefineConvToCore : public OpRewritePattern<refine::RefineConvOp> {
public:
    using OpRewritePattern<refine::RefineConvOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineConvOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        Value bias = op.getOperand(2);

        auto kernel_shape = op->getAttr("kernel_shape");
        auto padding = op->getAttr("padding");
        auto strides = op->getAttr("strides");
        auto ID = op->getAttr("ID");

        auto castID = ID.cast<StringAttr>();

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto twoAttr = rewriter.getI32IntegerAttr(2);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);
        
        //get out shape
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, input);
        Value kernelMemref = rewriter.create<core::CoreAllocOp>(loc, kernel);
        Value biasMemref = rewriter.create<core::CoreAllocOp>(loc, bias);

        auto inputMemrefType = inputMemref.getType().dyn_cast<mlir::MemRefType>();
        auto kernelMemrefType = inputMemref.getType().dyn_cast<mlir::MemRefType>();
        auto biasMemrefType = biasMemref.getType().dyn_cast<mlir::MemRefType>();

        llvm::ArrayRef<int64_t> _input_shape = inputMemrefType.getShape();
        llvm::ArrayRef<int64_t> _kernel_shape = kernelMemrefType.getShape();
        llvm::ArrayRef<int64_t> _bias_shape = biasMemrefType.getShape();

        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
        llvm::SmallVector<mlir::Attribute, 1> kernel_shape_attr;
        llvm::SmallVector<mlir::Attribute, 1> bias_shape_attr;



        for(int64_t dimSize : _input_shape){
            // Value dimValue = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        for(int64_t dimSize : _kernel_shape){
            // Value dimValue = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
            kernel_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        for(int64_t dimSize : _bias_shape){
            // Value dimValue = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);
        auto kernel_shape_arrayAttr = rewriter.getArrayAttr(kernel_shape_attr);
        auto bias_shape_arrayAttr = rewriter.getArrayAttr(bias_shape_attr);

        //input setup
        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, inputMemref, castID, zeroAttr, input_shape_arrayAttr); // arg0 = input

        //kernel setup
        Value kernelAddr = rewriter.create<core::CoreWriteOp>(loc, kernelMemref, castID, oneAttr, kernel_shape_arrayAttr);

        //bias setup
        Value biasAddr = rewriter.create<core::CoreWriteOp>(loc, biasMemref, castID, twoAttr, bias_shape_arrayAttr);


    //core start

        auto paddingAttr = padding.dyn_cast_or_null<mlir::ArrayAttr>();
        auto padding2Attr = paddingAttr.getValue()[0].dyn_cast<mlir::IntegerAttr>();

        auto strideAttr = strides.dyn_cast_or_null<mlir::ArrayAttr>();
        auto stride2Attr = strideAttr.getValue()[0].dyn_cast<mlir::IntegerAttr>();

        if(_input_shape[0] == -1 || _kernel_shape[0] == -1){
            llvm::outs() << "input memref shape is dynamic\n";
            return failure();
        }

        int64_t paddingValueInt = padding2Attr.getInt();
        int64_t strideValueInt = stride2Attr.getInt();

        int64_t outH = (_input_shape[2] + 2 * paddingValueInt - (_kernel_shape[2] - 1) - 1) / strideValueInt + 1;
        int64_t outW = (_input_shape[3] + 2 * paddingValueInt - (_kernel_shape[3] - 1) - 1) / strideValueInt + 1;

        auto outputShape = rewriter.getI64ArrayAttr({_input_shape[0], _kernel_shape[0], outH, outW});

        // auto outNValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(_input_shape[0]));
        // auto outCValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(_kernel_shape[0]));
        // auto outHValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outH));
        // auto outWValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outW));

        int32_t outsize = outH * outW * _input_shape[0] * _kernel_shape[0];
        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);

        auto kernelValue = rewriter.create<arith::ConstantOp>(loc, int8Type, rewriter.getI32IntegerAttr(_kernel_shape[3]));
        auto paddingValue = rewriter.create<arith::ConstantOp>(loc, int8Type, rewriter.getI32IntegerAttr(paddingValueInt));
        auto strideValue = rewriter.create<arith::ConstantOp>(loc, int8Type, rewriter.getI32IntegerAttr(strideValueInt));

        rewriter.create<core::CoreStartOp>(loc, 
                                    castID, 
                                    one, // one : optype = conv type 
                                    outsizeValue,
                                    kernelValue,    //config0
                                    paddingValue,   //config1
                                    strideValue,    //config2
                                    inputAddr,      //arg0
                                    kernelAddr,     //arg1
                                    biasAddr,       //arg2
                                    zero, zero, zero, zero, zero, zero//arg3 ~ arg8
                                    );

        rewriter.create<core::CoreWaitOp>(loc, castID);
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, outsizeValueAttr, castID, resAttr, outputShape);

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineSigmoidToCore : public OpRewritePattern<refine::RefineSigmoidOp> {
public:
    using OpRewritePattern<refine::RefineSigmoidOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSigmoidOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand();

        auto ID = op->getAttr("ID");

        auto castID = ID.cast<StringAttr>();

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto twoAttr = rewriter.getI32IntegerAttr(2);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, input);

        auto inputMemrefType = inputMemref.getType().dyn_cast<mlir::MemRefType>();
        llvm::ArrayRef<int64_t> input_shape = inputMemrefType.getShape();

        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
        for(int64_t dimSize : input_shape){
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);

        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, inputMemref, castID, zeroAttr, input_shape_arrayAttr);

        llvm::SmallVector<mlir::IntegerAttr, 1> outputShape;

        int outsize = 1;

        for(int i=0; i<input_shape.size(); i++){
            if(input_shape[i] == -1){
                llvm::outs() << "input memref shape is dynamic\n";
                return failure();
            }
            outsize *= input_shape[i];
            outputShape.push_back(rewriter.getI64IntegerAttr(input_shape[i]));
        }

        llvm::SmallVector<mlir::Attribute, 1> outputShapeAttr;
        for(auto attr : outputShape){
            outputShapeAttr.push_back(attr);
        }

        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);
        rewriter.create<core::CoreStartOp>(loc,
                                    castID,
                                    two,
                                    outsizeValue,
                                    zero, zero, zero, //config0~2
                                    inputAddr, //arg0
                                    zero, zero, zero, zero, zero, zero, zero, zero //arg1~8
                                    );

        rewriter.create<core::CoreWaitOp>(loc, castID);
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, outsizeValueAttr, castID, resAttr, rewriter.getArrayAttr(outputShapeAttr));

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineMulToCore : public OpRewritePattern<refine::RefineMulOp> {
public:
    using OpRewritePattern<refine::RefineMulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineMulOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value X = op.getOperand(0);
        Value Y = op.getOperand(1);

        auto ID = op->getAttr("ID");

        auto castID = ID.cast<StringAttr>();

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto twoAttr = rewriter.getI32IntegerAttr(2);
        auto threeAttr = rewriter.getI32IntegerAttr(3);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto three = rewriter.create<arith::ConstantOp>(loc, int32Type, threeAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);
        
        Value XMemref = rewriter.create<core::CoreAllocOp>(loc, X);
        Value YMemref = rewriter.create<core::CoreAllocOp>(loc, Y);
        
        auto XMemrefType = XMemref.getType().dyn_cast<mlir::MemRefType>();
        auto YMemrefType = YMemref.getType().dyn_cast<mlir::MemRefType>();

        llvm::ArrayRef<int64_t> X_shape = XMemrefType.getShape();
        llvm::ArrayRef<int64_t> Y_shape = YMemrefType.getShape();

        llvm::SmallVector<mlir::Attribute, 1> X_shape_attr;
        llvm::SmallVector<mlir::Attribute, 1> Y_shape_attr;

        for(int64_t dimSize : X_shape){
            X_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        for(int64_t dimSize : Y_shape){
            Y_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        auto X_shape_arrayAttr = rewriter.getArrayAttr(X_shape_attr);
        auto Y_shape_arrayAttr = rewriter.getArrayAttr(Y_shape_attr);

        Value XAddr = rewriter.create<core::CoreWriteOp>(loc, XMemref, castID, zeroAttr, X_shape_arrayAttr);
        Value YAddr = rewriter.create<core::CoreWriteOp>(loc, YMemref, castID, oneAttr, Y_shape_arrayAttr);


        llvm::SmallVector<mlir::IntegerAttr, 1> outputShape;

        int outsize = 1;

        for(int i=0; i<X_shape.size(); i++){
            if(X_shape[i] == -1){
                llvm::outs() << "input memref shape is dynamic\n";
                return failure();
            }
            outsize *= X_shape[i];
            outputShape.push_back(rewriter.getI64IntegerAttr(X_shape[i]));
        }

        llvm::SmallVector<mlir::Attribute, 1> outputShapeAttrs;
        for(auto attr : outputShape){
            outputShapeAttrs.push_back(attr);
        }


        auto outputShapeAttr = rewriter.getArrayAttr(outputShapeAttrs);

        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);

        rewriter.create<core::CoreStartOp>(loc,
                                    castID,
                                    three,
                                    outsizeValue,
                                    zero, zero, zero, //config0~2
                                    XAddr, //arg0
                                    YAddr, 
                                    zero, zero, zero, zero, zero, zero, zero //arg1~8
                                    );

        rewriter.create<core::CoreWaitOp>(loc, castID);
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, outsizeValueAttr, castID, resAttr, outputShapeAttr);

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

void RefineToCoreLoweringPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertRefineConvToCore, ConvertRefineSigmoidToCore, 
        ConvertRefineMulToCore>(&getContext());

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

std::unique_ptr<Pass> createRefineToCorePass() {
    return std::make_unique<RefineToCoreLoweringPass>();
}

std::unique_ptr<Pass> createRefineToCorePass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<RefineToCoreLoweringPass>(execNodesOnCpu);
}


} // namespace onnx_mlir








