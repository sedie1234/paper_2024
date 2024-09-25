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
    
void declareFunction(mlir::ModuleOp module, mlir::OpBuilder &builder);

namespace {

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


class ConvertRefineFusedConvToCore : public OpRewritePattern<refine::RefineFusedConvOp> {
public:
    using OpRewritePattern<refine::RefineFusedConvOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineFusedConvOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        // Value bias = op.getOperand(2);
        Value bias;

        // Check if bias operand exists
        if (op.getNumOperands() > 2) {
            bias = op.getOperand(2);  // If bias exists, use it
        } else {
            bias = mlir::Value();     // If bias does not exist, set bias as NoneType (empty Value)
        }

        auto kernel_shape = op->getAttr("kernel_shape");
        auto padding = op->getAttr("padding");
        auto strides = op->getAttr("strides");
        auto ID = op->getAttr("ID");

        auto castID = ID.cast<StringAttr>();

        auto f32Type = rewriter.getF32Type();
        auto llvmF32PtrType = LLVM::LLVMPointerType::get(f32Type);

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto twoAttr = rewriter.getI32IntegerAttr(2);
        auto nineAttr = rewriter.getI32IntegerAttr(9);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto nine = rewriter.create<arith::ConstantOp>(loc, int32Type, nineAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);
        
        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto kernelTensorType = kernel.getType().dyn_cast<mlir::RankedTensorType>();


        if(!inputTensorType || !kernelTensorType){
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }

        llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> kernelShape = kernelTensorType.getShape();

        mlir::Type inputElementType = inputTensorType.getElementType();
        mlir::Type kernelElementType = kernelTensorType.getElementType();

        auto inputMemrefType = mlir::MemRefType::get(inputShape, inputElementType);
        auto kernelMemrefType = mlir::MemRefType::get(kernelShape, kernelElementType);

        //get out shape
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);
        Value kernelMemref = rewriter.create<core::CoreAllocOp>(loc, kernelMemrefType, kernel);

        llvm::ArrayRef<int64_t> _input_shape = inputMemrefType.getShape();
        llvm::ArrayRef<int64_t> _kernel_shape = kernelMemrefType.getShape();

        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
        llvm::SmallVector<mlir::Attribute, 1> kernel_shape_attr;

        for(int64_t dimSize : _input_shape){
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        for(int64_t dimSize : _kernel_shape){
            kernel_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }


        llvm::SmallVector<mlir::Attribute, 1> bias_shape_attr;
        llvm::ArrayRef<int64_t> _bias_shape;
        Value biasMemref;


        if(bias){
            auto biasTensorType = bias.getType().dyn_cast<mlir::RankedTensorType>();
            llvm::ArrayRef<int64_t> biasShape = biasTensorType.getShape();
            mlir::Type biasElementType = biasTensorType.getElementType();
            
            auto biasMemrefType = mlir::MemRefType::get(biasShape, biasElementType);
            biasMemref = rewriter.create<core::CoreAllocOp>(loc, biasMemrefType, bias);


            
            _bias_shape = biasMemrefType.getShape();
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));


            for(int64_t dimSize : _bias_shape){
                bias_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
            }

        }else{
            auto biasMemrefType = mlir::MemRefType::get({_kernel_shape[0], 1, 1, 1}, f32Type);
            biasMemref = rewriter.create<memref::AllocOp>(loc, biasMemrefType);

            _bias_shape = biasMemrefType.getShape();
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(_kernel_shape[0]));
        }

        

        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);
        auto kernel_shape_arrayAttr = rewriter.getArrayAttr(kernel_shape_attr);
        auto bias_shape_arrayAttr = rewriter.getArrayAttr(bias_shape_attr);

        //input setup
        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, inputMemref, castID, zero, input_shape_arrayAttr); // arg0 = input

        //kernel setup
        Value kernelAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, kernelMemref, castID, one, kernel_shape_arrayAttr);

        //bias setup
        Value biasAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, biasMemref, castID, two, bias_shape_arrayAttr);

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

        int32_t outsize = outH * outW * _input_shape[0] * _kernel_shape[0];
        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);

        auto kernelValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(_kernel_shape[3]));
        auto paddingValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(paddingValueInt));
        auto strideValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(strideValueInt));

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, int32Type,
                                    castID, 
                                    nine, // nine : optype = fusedconv type 
                                    outsizeValue,
                                    kernelValue,    //config0
                                    paddingValue,   //config1
                                    strideValue,    //config2
                                    inputAddr,      //arg0
                                    kernelAddr,     //arg1
                                    biasAddr,       //arg2
                                    zero, zero, zero, zero, zero, zero,//arg3 ~ arg8
                                    inputAddr
                                    );

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShape, wait_chain_out.getResult());

        // rewriter.create<memref::DeallocOp>(loc, inputMemref);
        // rewriter.create<memref::DeallocOp>(loc, kernelMemref);
        // rewriter.create<memref::DeallocOp>(loc, biasMemref);

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};


class ConvertRefineConvToCore : public OpRewritePattern<refine::RefineConvOp> {
public:
    using OpRewritePattern<refine::RefineConvOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineConvOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand(0);
        Value kernel = op.getOperand(1);
        // Value bias = op.getOperand(2);
        Value bias;

        // Check if bias operand exists
        if (op.getNumOperands() > 2) {
            bias = op.getOperand(2);  // If bias exists, use it
        } else {
            bias = mlir::Value();     // If bias does not exist, set bias as NoneType (empty Value)
        }

        auto kernel_shape = op->getAttr("kernel_shape");
        auto padding = op->getAttr("padding");
        auto strides = op->getAttr("strides");
        auto ID = op->getAttr("ID");

        auto castID = ID.cast<StringAttr>();

        auto f32Type = rewriter.getF32Type();
        auto llvmF32PtrType = LLVM::LLVMPointerType::get(f32Type);

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
        
        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto kernelTensorType = kernel.getType().dyn_cast<mlir::RankedTensorType>();


        if(!inputTensorType || !kernelTensorType){
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }

        llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> kernelShape = kernelTensorType.getShape();

        mlir::Type inputElementType = inputTensorType.getElementType();
        mlir::Type kernelElementType = kernelTensorType.getElementType();

        auto inputMemrefType = mlir::MemRefType::get(inputShape, inputElementType);
        auto kernelMemrefType = mlir::MemRefType::get(kernelShape, kernelElementType);

        //get out shape
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);
        Value kernelMemref = rewriter.create<core::CoreAllocOp>(loc, kernelMemrefType, kernel);

        llvm::ArrayRef<int64_t> _input_shape = inputMemrefType.getShape();
        llvm::ArrayRef<int64_t> _kernel_shape = kernelMemrefType.getShape();

        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
        llvm::SmallVector<mlir::Attribute, 1> kernel_shape_attr;

        for(int64_t dimSize : _input_shape){
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        for(int64_t dimSize : _kernel_shape){
            kernel_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }


        llvm::SmallVector<mlir::Attribute, 1> bias_shape_attr;
        llvm::ArrayRef<int64_t> _bias_shape;
        Value biasMemref;


        if(bias){
            auto biasTensorType = bias.getType().dyn_cast<mlir::RankedTensorType>();
            llvm::ArrayRef<int64_t> biasShape = biasTensorType.getShape();
            mlir::Type biasElementType = biasTensorType.getElementType();
            
            auto biasMemrefType = mlir::MemRefType::get(biasShape, biasElementType);
            biasMemref = rewriter.create<core::CoreAllocOp>(loc, biasMemrefType, bias);


            
            _bias_shape = biasMemrefType.getShape();
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));


            for(int64_t dimSize : _bias_shape){
                bias_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
            }

        }else{
            auto biasMemrefType = mlir::MemRefType::get({_kernel_shape[0], 1, 1, 1}, f32Type);
            biasMemref = rewriter.create<memref::AllocOp>(loc, biasMemrefType);

            _bias_shape = biasMemrefType.getShape();
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(1));
            bias_shape_attr.push_back(rewriter.getI64IntegerAttr(_kernel_shape[0]));
        }

        

        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);
        auto kernel_shape_arrayAttr = rewriter.getArrayAttr(kernel_shape_attr);
        auto bias_shape_arrayAttr = rewriter.getArrayAttr(bias_shape_attr);

        //input setup
        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, inputMemref, castID, zero, input_shape_arrayAttr); // arg0 = input

        //kernel setup
        Value kernelAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, kernelMemref, castID, one, kernel_shape_arrayAttr);

        //bias setup
        Value biasAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, biasMemref, castID, two, bias_shape_arrayAttr);

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

        int32_t outsize = outH * outW * _input_shape[0] * _kernel_shape[0];
        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);

        auto kernelValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(_kernel_shape[3]));
        auto paddingValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(paddingValueInt));
        auto strideValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(strideValueInt));

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, int32Type,
                                    castID, 
                                    one, // one : optype = conv type 
                                    outsizeValue,
                                    kernelValue,    //config0
                                    paddingValue,   //config1
                                    strideValue,    //config2
                                    inputAddr,      //arg0
                                    kernelAddr,     //arg1
                                    biasAddr,       //arg2
                                    zero, zero, zero, zero, zero, zero,//arg3 ~ arg8
                                    inputAddr
                                    );

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShape, wait_chain_out.getResult());

        // rewriter.create<memref::DeallocOp>(loc, inputMemref);
        // rewriter.create<memref::DeallocOp>(loc, kernelMemref);
        // rewriter.create<memref::DeallocOp>(loc, biasMemref);

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineSigmoidToCore : public OpRewritePattern<refine::RefineSigmoidOp> {
public:
    using OpRewritePattern<refine::RefineSigmoidOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSigmoidOp op, PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

    Value input = op.getOperand();

    auto ID = op->getAttr("ID");
    auto castID = ID.cast<StringAttr>();

    auto int32Type = rewriter.getIntegerType(32);

    auto zeroAttr = rewriter.getI32IntegerAttr(0);
    auto oneAttr = rewriter.getI32IntegerAttr(1);
    auto twoAttr = rewriter.getI32IntegerAttr(2);
    auto resAttr = rewriter.getI32IntegerAttr(55);

    auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
    auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
    auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

    // 1. AllocOp 생성
    auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
    if (!inputTensorType) {
        return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
    }

    llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
    mlir::Type elementType = inputTensorType.getElementType();
    auto inputMemrefType = mlir::MemRefType::get(inputShape, elementType);

    Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);

    // 2. WriteOp 생성
    llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
    // llvm::outs() << "Refine Sigmoid input shape : " << castID << "\n";
    for (int64_t dimSize : inputShape) {
        // llvm::outs() << dimSize << ", ";
        input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
    }
    // llvm::outs() << "\n";

    auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);

    Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, inputMemref, castID, zero, input_shape_arrayAttr);

    // 3. StartOp 생성
    int outsize = 1;
    llvm::SmallVector<mlir::IntegerAttr, 1> outputShape;

    for (int i = 0; i < inputShape.size(); i++) {
        if (inputShape[i] == -1) {
            llvm::outs() << "input memref shape is dynamic\n";
            return failure();
        }
        outsize *= inputShape[i];
        outputShape.push_back(rewriter.getI64IntegerAttr(inputShape[i]));
    }

    outsize = 1;
    llvm::ArrayRef<int64_t> result_shape = resultType.getShape();

    // llvm::outs() << "Refine Sigmoid output shape : " << castID << "\n";
    for(int64_t dim : result_shape){
        outsize *= dim;
        // llvm::outs() << dim << ", ";
    }
    // llvm::outs() << "\n";

    auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
    auto start_chain_out = rewriter.create<core::CoreStartOp>(
        loc, int32Type, castID, two, outsizeValue, zero, zero, zero,  // config0~2
        inputAddr,  // arg0
        zero, zero, zero, zero, zero, zero, zero, zero,  // arg1~8
        inputAddr
    );

    // 4. WaitOp 생성
    auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());

    // 5. ReadOp 생성 및 최종 대체
    llvm::SmallVector<mlir::Attribute, 1> outputShapeAttr;
    for (auto attr : outputShape) {
        outputShapeAttr.push_back(attr);
    }

    auto outputTensor = rewriter.create<core::CoreReadOp>(
        loc, resultType, outsizeValue, castID, res, rewriter.getArrayAttr(outputShapeAttr), wait_chain_out.getResult()
    );

    // rewriter.create<memref::DeallocOp>(loc, inputMemref);

    // refine::RefineSigmoidOp을 ReadOp로 대체
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

        auto XTensorType = X.getType().dyn_cast<mlir::RankedTensorType>();
        auto YTensorType = Y.getType().dyn_cast<mlir::RankedTensorType>();

        if (!XTensorType || !YTensorType) {
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }
        
        llvm::ArrayRef<int64_t> XShape = XTensorType.getShape();
        llvm::ArrayRef<int64_t> YShape = YTensorType.getShape();

        mlir::Type XelementType = XTensorType.getElementType();
        mlir::Type YelementType = YTensorType.getElementType();

        auto XMemrefType = mlir::MemRefType::get(XShape, XelementType);
        auto YMemrefType = mlir::MemRefType::get(YShape, YelementType);

        Value XMemref = rewriter.create<core::CoreAllocOp>(loc, XMemrefType, X);
        Value YMemref = rewriter.create<core::CoreAllocOp>(loc, YMemrefType, Y);

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

        Value XAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, XMemref, castID, zero, X_shape_arrayAttr);
        Value YAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, YMemref, castID, one, Y_shape_arrayAttr);


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

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, 
                                                                int32Type,
                                                                castID,
                                                                three,
                                                                outsizeValue,
                                                                zero, zero, zero, //config0~2
                                                                XAddr, //arg0
                                                                YAddr, 
                                                                zero, zero, zero, zero, zero, zero, zero, //arg2~8
                                                                XAddr);

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShapeAttr, wait_chain_out.getResult());

        // rewriter.create<memref::DeallocOp>(loc, XMemref);
        // rewriter.create<memref::DeallocOp>(loc, YMemref);

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineSubToCore : public OpRewritePattern<refine::RefineSubOp> {
public:
    using OpRewritePattern<refine::RefineSubOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSubOp op, PatternRewriter &rewriter) const override{

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
        auto sevenAttr = rewriter.getI32IntegerAttr(7);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto three = rewriter.create<arith::ConstantOp>(loc, int32Type, threeAttr);
        auto seven = rewriter.create<arith::ConstantOp>(loc, int32Type, sevenAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

        auto XTensorType = X.getType().dyn_cast<mlir::RankedTensorType>();
        auto YTensorType = Y.getType().dyn_cast<mlir::RankedTensorType>();

        if (!XTensorType || !YTensorType) {
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }
        
        llvm::ArrayRef<int64_t> XShape = XTensorType.getShape();
        llvm::ArrayRef<int64_t> YShape = YTensorType.getShape();

        mlir::Type XelementType = XTensorType.getElementType();
        mlir::Type YelementType = YTensorType.getElementType();

        auto XMemrefType = mlir::MemRefType::get(XShape, XelementType);
        auto YMemrefType = mlir::MemRefType::get(YShape, YelementType);

        Value XMemref = rewriter.create<core::CoreAllocOp>(loc, XMemrefType, X);
        Value YMemref = rewriter.create<core::CoreAllocOp>(loc, YMemrefType, Y);

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

        Value XAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, XMemref, castID, zero, X_shape_arrayAttr);
        Value YAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, YMemref, castID, one, Y_shape_arrayAttr);


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

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, 
                                                                int32Type,
                                                                castID,
                                                                seven,
                                                                outsizeValue,
                                                                zero, zero, zero, //config0~2
                                                                XAddr, //arg0
                                                                YAddr, 
                                                                zero, zero, zero, zero, zero, zero, zero, //arg2~8
                                                                XAddr);

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShapeAttr, wait_chain_out.getResult());

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineDivToCore : public OpRewritePattern<refine::RefineDivOp> {
public:
    using OpRewritePattern<refine::RefineDivOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineDivOp op, PatternRewriter &rewriter) const override{

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
        auto eightAttr = rewriter.getI32IntegerAttr(8);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto three = rewriter.create<arith::ConstantOp>(loc, int32Type, threeAttr);
        auto eight = rewriter.create<arith::ConstantOp>(loc, int32Type, eightAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

        auto XTensorType = X.getType().dyn_cast<mlir::RankedTensorType>();
        auto YTensorType = Y.getType().dyn_cast<mlir::RankedTensorType>();

        if (!XTensorType || !YTensorType) {
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }
        
        llvm::ArrayRef<int64_t> XShape = XTensorType.getShape();
        llvm::ArrayRef<int64_t> YShape = YTensorType.getShape();

        mlir::Type XelementType = XTensorType.getElementType();
        mlir::Type YelementType = YTensorType.getElementType();

        auto XMemrefType = mlir::MemRefType::get(XShape, XelementType);
        auto YMemrefType = mlir::MemRefType::get(YShape, YelementType);

        Value XMemref = rewriter.create<core::CoreAllocOp>(loc, XMemrefType, X);
        Value YMemref = rewriter.create<core::CoreAllocOp>(loc, YMemrefType, Y);

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

        Value XAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, XMemref, castID, zero, X_shape_arrayAttr);
        Value YAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, YMemref, castID, one, Y_shape_arrayAttr);


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

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, 
                                                                int32Type,
                                                                castID,
                                                                eight,
                                                                outsizeValue,
                                                                zero, zero, zero, //config0~2
                                                                XAddr, //arg0
                                                                YAddr, 
                                                                zero, zero, zero, zero, zero, zero, zero, //arg2~8
                                                                XAddr);

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShapeAttr, wait_chain_out.getResult());

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineSplitToCore : public OpRewritePattern<refine::RefineSplitOp> {
public:
    using OpRewritePattern<refine::RefineSplitOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSplitOp op, PatternRewriter &rewriter) const override{
        auto loc = op.getLoc();

        auto f32Type = rewriter.getF32Type();
        auto int32Type = rewriter.getIntegerType(32);
        auto int64Type = rewriter.getIntegerType(64);
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);


        Value input = op.getOperand(0);
        Value split = op.getOperand(1);

        auto output0 = op.getResult(0);
        auto output1 = op.getResult(1);

        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto output0TensorType = output0.getType().dyn_cast<mlir::RankedTensorType>();
        auto output1TensorType = output1.getType().dyn_cast<mlir::RankedTensorType>();

        //get shape
        llvm::ArrayRef<int64_t> inputshape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> output0shape = output0TensorType.getShape();
        llvm::ArrayRef<int64_t> output1shape = output1TensorType.getShape();

        //shape info mem
        auto shapeLength = rewriter.getIntegerAttr(int64Type, inputshape.size());
        auto shapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shapeLength);
        Value shapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, shapeLengthValue, 0);

        // save the shape info
        for (int64_t i = 0; i < inputshape.size(); i++) {
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, shapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(inputshape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        auto symbolRefSplit = SymbolRefAttr::get(rewriter.getContext(), "runtimeSplit");

        //declare memreftypes
        auto inputMemrefType = MemRefType::get(inputshape, f32Type);
        auto out0MemrefType = MemRefType::get(output0shape, f32Type);
        auto out1MemrefType = MemRefType::get(output1shape, f32Type);

        //input tensor to memref
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);
        
        //get input pointer
        Value idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputMemref);
        Value idxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx);
        Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, idxInt);

        //alloc output0, output1
        Value out0Mem = rewriter.create<memref::AllocOp>(loc, out0MemrefType);
        Value out1Mem = rewriter.create<memref::AllocOp>(loc, out1MemrefType);

        //get out0, out1 mem pointer
        Value out0Idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out0Mem);
        Value out1Idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out1Mem);
        Value out0IdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, out0Idx);
        Value out1IdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, out1Idx);
        Value out0Ptr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, out0IdxInt);
        Value out1Ptr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, out1IdxInt);

        Value index0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value index1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value value0 = rewriter.create<tensor::ExtractOp>(loc, split, index0);
        Value value1 = rewriter.create<tensor::ExtractOp>(loc, split, index1);

        // runtime call
        rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                                         symbolRefSplit,
                                                         ValueRange{ptr, out0Ptr, out1Ptr, shapeAlloca, 
                                                         value0, value1, shapeLengthValue});

        Value out0Tensor = rewriter.create<UnrealizedConversionCastOp>(loc, output0TensorType, out0Mem).getResult(0);
        Value out1Tensor = rewriter.create<UnrealizedConversionCastOp>(loc, output1TensorType, out1Mem).getResult(0);

        rewriter.replaceOp(op, {out0Tensor, out1Tensor});

    }
};

class ConvertRefineAddToCore : public OpRewritePattern<refine::RefineAddOp> {
public:
    using OpRewritePattern<refine::RefineAddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineAddOp op, PatternRewriter &rewriter) const override{

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
        auto fourAttr = rewriter.getI32IntegerAttr(4);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto three = rewriter.create<arith::ConstantOp>(loc, int32Type, threeAttr);
        auto four = rewriter.create<arith::ConstantOp>(loc, int32Type, fourAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

        auto XTensorType = X.getType().dyn_cast<mlir::RankedTensorType>();
        auto YTensorType = Y.getType().dyn_cast<mlir::RankedTensorType>();

        if (!XTensorType || !YTensorType) {
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }
        
        llvm::ArrayRef<int64_t> XShape = XTensorType.getShape();
        llvm::ArrayRef<int64_t> YShape = YTensorType.getShape();

        mlir::Type XelementType = XTensorType.getElementType();
        mlir::Type YelementType = YTensorType.getElementType();

        auto XMemrefType = mlir::MemRefType::get(XShape, XelementType);
        auto YMemrefType = mlir::MemRefType::get(YShape, YelementType);

        Value XMemref = rewriter.create<core::CoreAllocOp>(loc, XMemrefType, X);
        Value YMemref = rewriter.create<core::CoreAllocOp>(loc, YMemrefType, Y);

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

        Value XAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, XMemref, castID, zero, X_shape_arrayAttr);
        Value YAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, YMemref, castID, one, Y_shape_arrayAttr);


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

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, 
                                                                int32Type,
                                                                castID,
                                                                four,
                                                                outsizeValue,
                                                                zero, zero, zero, //config0~2
                                                                XAddr, //arg0
                                                                YAddr, 
                                                                zero, zero, zero, zero, zero, zero, zero, //arg2~8
                                                                XAddr);

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShapeAttr, wait_chain_out.getResult());

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineConcatToCore : public OpRewritePattern<refine::RefineConcatOp> {
public:
    using OpRewritePattern<refine::RefineConcatOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineConcatOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        auto f32Type = rewriter.getF32Type();
        auto int32Type = rewriter.getIntegerType(32);
        auto int64Type = rewriter.getIntegerType(64);
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);

        Type resultType = op.getResult().getType();

        Value input0 = op.getOperand(0);
        Value input1 = op.getOperand(1);

        auto output = op.getResult();

        auto input0TensorType = input0.getType().dyn_cast<mlir::RankedTensorType>();
        auto input1TensorType = input1.getType().dyn_cast<mlir::RankedTensorType>();
        auto outputTensorType = output.getType().dyn_cast<mlir::RankedTensorType>();

        llvm::ArrayRef<int64_t> input0Shape = input0TensorType.getShape();
        llvm::ArrayRef<int64_t> input1Shape = input1TensorType.getShape();
        llvm::ArrayRef<int64_t> outputShape = outputTensorType.getShape();

        auto shape0Length = rewriter.getIntegerAttr(int64Type, 4);
        auto shape1Length = rewriter.getIntegerAttr(int64Type, 4);
        auto shape0LengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shape0Length);
        auto shape1LengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shape1Length);
        Value shape0Alloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, shape0LengthValue, 0);
        Value shape1Alloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, shape1LengthValue, 0);

        for(int64_t i=0; i< input0Shape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, shape0Alloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(input0Shape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        for(int64_t i=0; i< input1Shape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, shape1Alloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(input1Shape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        auto symbolRefConcat = SymbolRefAttr::get(rewriter.getContext(), "runtimeConcat");

        auto input0MemrefType = MemRefType::get(input0Shape, f32Type);
        auto input1MemrefType = MemRefType::get(input1Shape, f32Type);
        auto outputMemrefType = MemRefType::get(outputShape, f32Type);

        Value input0Memref = rewriter.create<core::CoreAllocOp>(loc, input0MemrefType, input0);
        Value input1Memref = rewriter.create<core::CoreAllocOp>(loc, input1MemrefType, input1);

        Value idx0 = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input0Memref);
        Value idx1 = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input1Memref);

        Value idx0Int = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx0);
        Value idx1Int = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx1);

        Value ptr0 = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, idx0Int);
        Value ptr1 = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, idx1Int);

        Value outMem = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
        Value outIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outMem);
        Value outIdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, outIdx);
        Value outPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, outIdxInt);

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefConcat,
                                      ValueRange{ptr0, ptr1, outPtr, shape0Alloca, shape1Alloca});

        Value outputTensor = rewriter.create<UnrealizedConversionCastOp>(loc, outputTensorType, outMem).getResult(0);
        rewriter.replaceOp(op, outputTensor);

    }
};

class ConvertRefineMaxpoolToCore : public OpRewritePattern<refine::RefineMaxpoolOp> {
public:
    using OpRewritePattern<refine::RefineMaxpoolOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineMaxpoolOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();
        Type resultType = op.getResult().getType();

        Value input = op.getOperand();

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
        auto fiveAttr = rewriter.getI32IntegerAttr(5);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto one = rewriter.create<arith::ConstantOp>(loc, int32Type, oneAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto five = rewriter.create<arith::ConstantOp>(loc, int32Type, fiveAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);
        
        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();

        if(!inputTensorType){
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }

        llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
        mlir::Type inputElementType = inputTensorType.getElementType();
        auto inputMemrefType = mlir::MemRefType::get(inputShape, inputElementType);

        //get out shape
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);
        llvm::ArrayRef<int64_t> _input_shape = inputMemrefType.getShape();
        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;

        for(int64_t dimSize : _input_shape){
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }

        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);

        //input setup
        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, inputMemref, castID, zero, input_shape_arrayAttr); // arg0 = input

    //core start
        auto kernel_shape_attr = kernel_shape.dyn_cast_or_null<mlir::ArrayAttr>();
        auto _kernel_shape = kernel_shape_attr.getValue()[0].dyn_cast<mlir::IntegerAttr>();

        auto paddingAttr = padding.dyn_cast_or_null<mlir::ArrayAttr>();
        auto padding2Attr = paddingAttr.getValue()[0].dyn_cast<mlir::IntegerAttr>();

        auto strideAttr = strides.dyn_cast_or_null<mlir::ArrayAttr>();
        auto stride2Attr = strideAttr.getValue()[0].dyn_cast<mlir::IntegerAttr>();

        if(_input_shape[0] == -1){
            llvm::outs() << "input memref shape is dynamic\n";
            return failure();
        }

        int64_t paddingValueInt = padding2Attr.getInt();
        int64_t strideValueInt = stride2Attr.getInt();
        int64_t kernelValueInt = _kernel_shape.getInt();

        int64_t outH = (_input_shape[2] + 2 * paddingValueInt - kernelValueInt) / strideValueInt + 1;
        int64_t outW = (_input_shape[3] + 2 * paddingValueInt - kernelValueInt) / strideValueInt + 1;

        auto outputShape = rewriter.getI64ArrayAttr({_input_shape[0], _input_shape[1], outH, outW});

        int32_t outsize = outH * outW * _input_shape[0] * _input_shape[1];
        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto outsizeValueAttr = rewriter.getI32IntegerAttr(outsize);

        auto kernelValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(kernelValueInt));
        auto paddingValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(paddingValueInt));
        auto strideValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(strideValueInt));

        auto start_chain_out = rewriter.create<core::CoreStartOp>(loc, int32Type,
                                    castID, 
                                    five, // five : optype = maxpool type 
                                    outsizeValue,
                                    kernelValue,    //config0
                                    paddingValue,   //config1
                                    strideValue,    //config2
                                    inputAddr,      //arg0
                                    zero, zero, zero, zero, zero, zero, zero, zero,//arg1 ~ arg8
                                    inputAddr
                                    );

        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());
        auto outputTensor = rewriter.create<core::CoreReadOp>(loc, resultType, outsizeValue, castID, res, outputShape, wait_chain_out.getResult());

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class ConvertRefineResizeToCore : public OpRewritePattern<refine::RefineResizeOp>{
public:
    using OpRewritePattern<refine::RefineResizeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineResizeOp op, PatternRewriter &rewriter) const override{
        auto loc = op.getLoc();

        auto f32Type = rewriter.getF32Type();
        auto int32Type = rewriter.getIntegerType(32);
        auto int64Type = rewriter.getIntegerType(64);
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);

        auto input = op.getOperand();
        auto output = op.getResult();

        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto outputTensorType = output.getType().dyn_cast<mlir::RankedTensorType>();

        llvm::ArrayRef<int64_t> inputshape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> outputshape = outputTensorType.getShape();

        //shpae info mem
        auto shapeLength = rewriter.getIntegerAttr(int64Type, 4);
        auto shapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shapeLength);
        Value shapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, shapeLengthValue, 0);

        //save the shape info
        for(int64_t i=0; i<inputshape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, shapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(inputshape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        auto symbolRefResize = SymbolRefAttr::get(rewriter.getContext(), "runtimeResize");

        auto inputMemrefType = MemRefType::get(inputshape, f32Type);
        auto outputMemrefType = MemRefType::get(outputshape, f32Type);

        //input tensor to memref
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);

        //get input pointer
        Value idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputMemref);
        Value idxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx);
        Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, idxInt);

        //alloc output
        Value outMem = rewriter.create<memref::AllocOp>(loc, outputMemrefType);

        //get out mem pointer
        Value outIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outMem);
        Value outIdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, outIdx);
        Value outPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, outIdxInt);

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefResize, ValueRange{ptr, outPtr, shapeAlloca});

        Value outTensor = rewriter.create<UnrealizedConversionCastOp>(loc, outputTensorType, outMem).getResult(0);

        rewriter.replaceOp(op, outTensor);

    }

};

class ConvertRefineTransposeToCore : public OpRewritePattern<refine::RefineTransposeOp>{
public:
    using OpRewritePattern<refine::RefineTransposeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineTransposeOp op, PatternRewriter &rewriter) const override{
        auto loc = op.getLoc();

        auto f32Type = rewriter.getF32Type();
        auto int32Type = rewriter.getIntegerType(32);
        auto int64Type = rewriter.getIntegerType(64);
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);

        auto input = op.getOperand();
        auto output = op.getResult();

        auto permAttr = op->getAttr("perm").cast<ArrayAttr>();

        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto outputTensorType = output.getType().dyn_cast<mlir::RankedTensorType>();

        //get int value of perm info
        llvm::SmallVector<int64_t, 4> perm;
        for(auto attr : permAttr.getValue()){
            auto intAttr = attr.dyn_cast<mlir::IntegerAttr>();
            perm.push_back(intAttr.getInt());
        }

        auto permLength = rewriter.getIntegerAttr(int64Type, 4);
        auto permLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, permLength);
        auto permAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, permLengthValue, 0);

        //save the perm info
        for(int64_t i=0; i<perm.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, permAlloca, ValueRange{idxI64});
            Value permElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(perm[i]));
            rewriter.create<LLVM::StoreOp>(loc, permElement, _ptr);
        }

        //input output shape set
        llvm::ArrayRef<int64_t> inputshape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> outputshape = outputTensorType.getShape();
 
        //shape info mem
        auto shapeLength = rewriter.getIntegerAttr(int64Type, 4);
        auto shapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shapeLength);
        Value shapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, shapeLengthValue, 0);

        //save the shape info
        for(int64_t i=0; i<inputshape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, shapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(inputshape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        auto symbolRefTranspose = SymbolRefAttr::get(rewriter.getContext(), "runtimeTranspose");

        auto inputMemrefType = MemRefType::get(inputshape, f32Type);
        auto outputMemrefType = MemRefType::get(outputshape, f32Type);

        //input tensor to memref
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);

        //get input pointer
        Value idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputMemref);
        Value idxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx);
        Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, idxInt);

        //alloc output
        Value outMem = rewriter.create<memref::AllocOp>(loc, outputMemrefType);

        //get out mem pointer
        Value outIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outMem);
        Value outIdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, outIdx);
        Value outPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, outIdxInt);

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefTranspose, ValueRange{ptr, outPtr, shapeAlloca, permAlloca});

        Value outTensor = rewriter.create<UnrealizedConversionCastOp>(loc, outputTensorType, outMem).getResult(0);

        rewriter.replaceOp(op, outTensor);

    }

};


class ConvertRefineSoftmaxToCore : public OpRewritePattern<refine::RefineSoftmaxOp>{
public:
    using OpRewritePattern<refine::RefineSoftmaxOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSoftmaxOp op, PatternRewriter &rewriter) const override{
        auto loc = op.getLoc();
        auto resultType = op.getResult().getType();

        Value input = op.getOperand();
        auto ID = op->getAttr("ID");
        auto castID = ID.cast<StringAttr>();

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);

        auto zeroAttr = rewriter.getI32IntegerAttr(0);
        auto oneAttr = rewriter.getI32IntegerAttr(1);
        auto twoAttr = rewriter.getI32IntegerAttr(2);
        auto sixAttr = rewriter.getI32IntegerAttr(6);
        auto resAttr = rewriter.getI32IntegerAttr(55);

        auto zero = rewriter.create<arith::ConstantOp>(loc, int32Type, zeroAttr);
        auto two = rewriter.create<arith::ConstantOp>(loc, int32Type, twoAttr);
        auto six = rewriter.create<arith::ConstantOp>(loc, int32Type, sixAttr);
        auto res = rewriter.create<arith::ConstantOp>(loc, int32Type, resAttr);

        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        if (!inputTensorType) {
            return rewriter.notifyMatchFailure(op, "Expected input to be a RankedTensorType");
        }

        llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
        mlir::Type elementType = inputTensorType.getElementType();
        auto inputMemrefType = mlir::MemRefType::get(inputShape, elementType);

        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);

        // 2. WriteOp 생성
        llvm::SmallVector<mlir::Attribute, 1> input_shape_attr;
        for (int64_t dimSize : inputShape) {
            input_shape_attr.push_back(rewriter.getI64IntegerAttr(dimSize));
        }
        auto input_shape_arrayAttr = rewriter.getArrayAttr(input_shape_attr);

        Value inputAddr = rewriter.create<core::CoreWriteOp>(loc, int32Type, inputMemref, castID, zero, input_shape_arrayAttr);

        // 3. StartOp 생성
        int outsize = 1;
        llvm::SmallVector<mlir::IntegerAttr, 1> outputShape;
        for (int i = 0; i < inputShape.size(); i++) {
            if (inputShape[i] == -1) {
                llvm::outs() << "input memref shape is dynamic\n";
                return failure();
            }
            outsize *= inputShape[i];
            outputShape.push_back(rewriter.getI64IntegerAttr(inputShape[i]));
        }



        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(outsize));
        auto start_chain_out = rewriter.create<core::CoreStartOp>(
            loc, int32Type, castID, six, outsizeValue, zero, zero, zero,  // config0~2
            inputAddr,  // arg0
            zero, zero, zero, zero, zero, zero, zero, zero,  // arg1~8
            inputAddr
        );

        // 4. WaitOp 생성
        auto wait_chain_out = rewriter.create<core::CoreWaitOp>(loc, int32Type, castID, start_chain_out.getResult());

        // 5. ReadOp 생성 및 최종 대체
        llvm::SmallVector<mlir::Attribute, 1> outputShapeAttr;
        for (auto attr : outputShape) {
            outputShapeAttr.push_back(attr);
        }

        auto outputTensor = rewriter.create<core::CoreReadOp>(
            loc, resultType, outsizeValue, castID, res, rewriter.getArrayAttr(outputShapeAttr), wait_chain_out.getResult()
        );

        // refine::RefineSigmoidOp을 ReadOp로 대체
        rewriter.replaceOp(op, outputTensor);

        return success();


    }
};

class ConvertRefineSliceToCore : public OpRewritePattern<refine::RefineSliceOp>{

public:
    using OpRewritePattern<refine::RefineSliceOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(refine::RefineSliceOp op, PatternRewriter &rewriter)const override{
        auto loc = op.getLoc();

        auto f32Type = rewriter.getF32Type();
        auto int32Type = rewriter.getIntegerType(32);
        auto int64Type = rewriter.getIntegerType(64);
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);

        Value input = op.getOperand(0);
        Value starts = op.getOperand(1);
        Value ends = op.getOperand(2);
        Value axes = op.getOperand(3);
        Value steps = op.getOperand(4);

        auto output = op.getResult();

        //get tensor type
        auto inputTensorType = input.getType().dyn_cast<mlir::RankedTensorType>();
        auto startsTensorType = starts.getType().dyn_cast<mlir::RankedTensorType>();
        auto endsTensorType = ends.getType().dyn_cast<mlir::RankedTensorType>();;
        auto axesTensorType = axes.getType().dyn_cast<mlir::RankedTensorType>();
        auto stepsTensorType = steps.getType().dyn_cast<mlir::RankedTensorType>();
        auto outputTensorType = output.getType().dyn_cast<mlir::RankedTensorType>();

        //get tensor shape
        llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
        llvm::ArrayRef<int64_t> startsShape = startsTensorType.getShape();
        llvm::ArrayRef<int64_t> endsShape = endsTensorType.getShape();
        llvm::ArrayRef<int64_t> axesShape = axesTensorType.getShape();
        llvm::ArrayRef<int64_t> stepsShape = stepsTensorType.getShape();
        llvm::ArrayRef<int64_t> outputShape = outputTensorType.getShape();

        //get memref type
        auto inputMemrefType = MemRefType::get(inputShape, f32Type);
        auto startsMemrefType = MemRefType::get(startsShape, int64Type);
        auto endsMemrefType = MemRefType::get(endsShape, int64Type);
        auto axesMemrefType = MemRefType::get(axesShape, int64Type);
        auto stepsMemrefType = MemRefType::get(stepsShape, int64Type);
        auto outputMemrefType = MemRefType::get(outputShape, f32Type);

        //arguments mem alloc
        Value inputMemref = rewriter.create<core::CoreAllocOp>(loc, inputMemrefType, input);
        Value startsMemref = rewriter.create<core::CoreAllocOp>(loc, startsMemrefType, starts);
        Value endsMemref = rewriter.create<core::CoreAllocOp>(loc, endsMemrefType, ends);
        Value axesMemref = rewriter.create<core::CoreAllocOp>(loc, axesMemrefType, axes);
        Value stepsMemref = rewriter.create<core::CoreAllocOp>(loc, stepsMemrefType, steps);

        //runtime function symbol
        auto symbolRefSlice = SymbolRefAttr::get(rewriter.getContext(), "runtimeSlice");

        //get index
        Value inputIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, inputMemref);
        Value startsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, startsMemref);
        Value endsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, endsMemref);
        Value axesIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, axesMemref);
        Value stepsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, stepsMemref);

        //get pointer int
        Value inputInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, inputIdx);
        Value startsInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, startsIdx);
        Value endsInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, endsIdx);
        Value axesInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, axesIdx);
        Value stepsInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, stepsIdx);

        //get pointer
        Value inputPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, inputInt);
        Value startsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, int64PtrType, startsInt);
        Value endsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, int64PtrType, endsInt);
        Value axesPtr = rewriter.create<LLVM::IntToPtrOp>(loc, int64PtrType, axesInt);
        Value stepsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, int64PtrType, stepsInt);
        
        //output pointer set
        Value outMem = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
        Value outIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outMem);
        Value outInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, outIdx);
        Value outPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, outInt);

        //get rank
        int64_t rank = startsTensorType.getRank();
        Value rankValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(rank));

        //get input output shape
        auto inputShapeLength = rewriter.getIntegerAttr(int64Type, 4);
        auto outputShapeLength = rewriter.getIntegerAttr(int64Type, 4);

        auto inputShapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, inputShapeLength);
        auto outputShapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, outputShapeLength);

        Value inputShapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, inputShapeLengthValue, 0);
        Value outputShapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, int64PtrType, outputShapeLengthValue, 0);

        for(int64_t i=0; i<inputShape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, inputShapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(inputShape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        for(int64_t i=0; i<outputShape.size(); i++){
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, int64PtrType, outputShapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(outputShape[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefSlice,
                                ValueRange{inputPtr, startsPtr, endsPtr, axesPtr, stepsPtr, outPtr, 
                                            rankValue, inputShapeAlloca, outputShapeAlloca});
        
        Value outputTensor = rewriter.create<UnrealizedConversionCastOp>(loc, outputTensorType, outMem).getResult(0);
        rewriter.replaceOp(op, outputTensor);
    }

};

void RefineToCoreLoweringPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp, core::CoreStartOp, core::CoreWriteOp, 
        core::CoreWaitOp, core::CoreReadOp, core::CoreAllocOp, memref::DeallocOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertRefineFusedConvToCore, ConvertRefineConvToCore, ConvertRefineSigmoidToCore, 
        ConvertRefineMulToCore, ConvertRefineSplitToCore, ConvertRefineAddToCore,
        ConvertRefineConcatToCore, ConvertRefineMaxpoolToCore, ConvertRefineResizeToCore,
        ConvertRefineTransposeToCore, ConvertRefineSoftmaxToCore, ConvertRefineSliceToCore,
        ConvertRefineSubToCore, ConvertRefineDivToCore>(&getContext());

    OpBuilder builder(module.getContext());
    declareFunction(module, builder);
    

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

std::unique_ptr<Pass> createRefineToCorePass() {
    return std::make_unique<RefineToCoreLoweringPass>();
}

std::unique_ptr<Pass> createRefineToCorePass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<RefineToCoreLoweringPass>(execNodesOnCpu);
}


void declareFunction(mlir::ModuleOp module, mlir::OpBuilder &builder){

    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto voidType = LLVM::LLVMVoidType::get(context);
    auto f32Type = FloatType::getF32(context);
    auto int64Type = IntegerType::get(context, 64);
    auto f32PtrType = LLVM::LLVMPointerType::get(f32Type);
    auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);
    auto int32Type = IntegerType::get(context, 32);
    auto charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context,8));

//runtimeSplit

    auto runtimeSplitFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, f32PtrType, f32PtrType, int64PtrType, 
                                    int64Type, int64Type, int32Type},
        false
    );

    auto runtimeConcatFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, f32PtrType, f32PtrType, int64PtrType, int64PtrType},
        false
    );

    auto runtimeResizeFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, f32PtrType, int64PtrType},
        false
    );

    auto runtimeTransposeFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, f32PtrType, int64PtrType, int64PtrType},
        false
    );

    auto runtimeSliceFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, int64PtrType, int64PtrType, int64PtrType, int64PtrType, f32PtrType, 
                                int32Type, int64PtrType, int64PtrType},
        false
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("runtimeSplit")){
        auto runtimeSplitFunc = builder.create<LLVM::LLVMFuncOp>(loc, "runtimeSplit", runtimeSplitFuncType);
        module.push_back(runtimeSplitFunc);
    }

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("runtimeConcat")){
        auto runtimeConcatFunc = builder.create<LLVM::LLVMFuncOp>(loc, "runtimeConcat", runtimeConcatFuncType);
        module.push_back(runtimeConcatFunc);
    }

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("runtimeResize")){
        auto runtimeResizeFunc = builder.create<LLVM::LLVMFuncOp>(loc, "runtimeResize", runtimeResizeFuncType);
        module.push_back(runtimeResizeFunc);
    }
    
    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("runtimeTranspose")){
        auto runtimeTransposeFunc = builder.create<LLVM::LLVMFuncOp>(loc, "runtimeTranspose", runtimeTransposeFuncType);
        module.push_back(runtimeTransposeFunc);
    }

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("runtimeSlice")){
        auto runtimeSliceFunc = builder.create<LLVM::LLVMFuncOp>(loc, "runtimeSlice", runtimeSliceFuncType);
        module.push_back(runtimeSliceFunc);
    }
}

} // namespace onnx_mlir








