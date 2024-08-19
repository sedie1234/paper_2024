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

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm-c/Core.h"

using namespace mlir;
namespace onnx_mlir {
    
void declareExternalFunction(mlir::ModuleOp module, mlir::OpBuilder &builder);

namespace {

#include "src/Accelerators/PA/Conversion/CoreToMLIR/ONNXCoreToMLIR.inc"


struct CoreToMLIRLoweringPass
    : public PassWrapper<CoreToMLIRLoweringPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoreToMLIRLoweringPass)

    StringRef getArgument() const override {return "convert-core-to-mlir";}
    StringRef getDescription() const override {
        return "Lower Refine ops to Core ops";
    }

    CoreToMLIRLoweringPass() = default;
    CoreToMLIRLoweringPass(const CoreToMLIRLoweringPass &pass)
        : PassWrapper<CoreToMLIRLoweringPass, OperationPass<ModuleOp>>() {}
    CoreToMLIRLoweringPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnSCpu",
        llvm::cl::desc("Core To MLIR Lowering Pass"),
        llvm::cl::ZeroOrMore};

};
} // end anonymous namespace

class ConvertCoreAllocToMLIR : public OpRewritePattern<core::CoreAllocOp> {
public:
    using OpRewritePattern<core::CoreAllocOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreAllocOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        Value inputTensor = op.getOperand();

        auto tensorType = inputTensor.getType().cast<mlir::RankedTensorType>();

        auto memrefType = mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        auto result = rewriter.create<UnrealizedConversionCastOp>(loc, memrefType, inputTensor).getResult(0);

        rewriter.replaceOp(op, result);

        return success();
    }
};

class ConvertCoreWriteToMLIR : public OpRewritePattern<core::CoreWriteOp> {
public:
    using OpRewritePattern<core::CoreWriteOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreWriteOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        auto int64Type = rewriter.getIntegerType(64);
        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto llvmInt64PtrType = LLVM::LLVMPointerType::get(int64Type);
        auto i8PtrType = LLVM::LLVMPointerType::get(int8Type);
        auto llvmF32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        
        
        Value input = op.getOperand(0);
        
        auto shapeAttr = op->getAttr("shape").cast<ArrayAttr>();

        auto ID = op->getAttr("ID");
        auto arg = op->getAttr("arg");

        auto castID = ID.cast<StringAttr>();
        auto castArg = arg.cast<IntegerAttr>();

        auto argValue = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg);

    
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

        Value idx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
        Value idxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, idx);
        Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmF32PtrType, idxInt);

        auto symbolRefWrite = SymbolRefAttr::get(rewriter.getContext(), "writeToAccel");

        auto memrefType = input.getType().dyn_cast<mlir::MemRefType>();
        llvm::ArrayRef<int64_t> shape = memrefType.getShape();

        int64_t size = 1;

        for(int i=0; i<shape.size(); i++){
            if(shape[i] == -1){
                llvm::outs() << "input memref shape is dynamic\n";
                return failure();
            }
            size *= shape[i];
        }
        auto sizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(size));
        

        mlir::SmallVector<int64_t, 4> shapeVector;
        for(auto attr : shapeAttr.getValue()){
            auto intAttr = attr.dyn_cast<mlir::IntegerAttr>();
            shapeVector.push_back(intAttr.getInt());
        }


        auto shapeMemRefType = MemRefType::get(shapeVector, rewriter.getIntegerType(64));
        Value shapeMemRef = rewriter.create<memref::AllocOp>(loc, shapeMemRefType);
        
        Value shapePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, shapeMemRef);
        Value shapePtrInt = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIntegerType(64), shapePtr);
        Value shapePtrCast = rewriter.create<LLVM::IntToPtrOp>(loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(64)), shapePtrInt);

        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);
        //get id pointer
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
        // auto alignmentValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(0));
        auto strAlloca = rewriter.create<LLVM::AllocaOp>(loc, i8PtrType, strLengthValue, 0);
            //store str to int8* format
        for(size_t i=0; i<idStr.size(); i++){
            auto charValue = rewriter.getI8IntegerAttr(idStr[i]);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(i));
            auto charPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{indexValue});
            rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), charValue), charPtr);
        }

        // null terminator
        auto nullTerminatorIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(idStr.size()));
        auto nullTerminatorPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{nullTerminatorIndex});
        rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(0)), nullTerminatorPtr);

        

        auto symbolRefWriteToAccel = SymbolRefAttr::get(rewriter.getContext(), "writeToAccel");
        // Value output = rewriter.create<LLVM::AllocaOp>(loc, llvmF32PtrType, sizeValue);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefWriteToAccel, 
            ValueRange{ ptr, sizeValue, strAlloca, argValue, shapePtrCast});

        return success();
    }
};

class ConvertCoreStartToMLIR : public OpRewritePattern<core::CoreStartOp> {
public:
    using OpRewritePattern<core::CoreStartOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreStartOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        auto i8PtrType = LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));
        auto int64Type = rewriter.getIntegerType(64);
        auto int32Type = rewriter.getIntegerType(32);
        auto f32Type = rewriter.getF32Type();
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        
        auto ID = op->getAttr("ID");
        auto optype = op->getAttr("optype");
        auto outsize = op->getAttr("outsize");
        auto config0 = op->getAttr("config0");
        auto config1 = op->getAttr("config1");
        auto config2 = op->getAttr("config2");
        auto arg0 = op->getAttr("arg0");
        auto arg1 = op->getAttr("arg1");
        auto arg2 = op->getAttr("arg2");
        auto arg3 = op->getAttr("arg3");
        auto arg4 = op->getAttr("arg4");
        auto arg5 = op->getAttr("arg5");
        auto arg6 = op->getAttr("arg6");
        auto arg7 = op->getAttr("arg7");
        auto arg8 = op->getAttr("arg8");

        auto castID = ID.cast<StringAttr>();
        auto castOpType = optype.cast<IntegerAttr>();
        auto castOutsize = outsize.cast<IntegerAttr>();
        auto castConfig0 = config0.cast<IntegerAttr>();
        auto castConfig1 = config1.cast<IntegerAttr>();
        auto castConfig2 = config2.cast<IntegerAttr>();
        auto castArg0 = arg0.cast<IntegerAttr>();
        auto castArg1 = arg1.cast<IntegerAttr>();
        auto castArg2 = arg2.cast<IntegerAttr>();
        auto castArg3 = arg3.cast<IntegerAttr>();
        auto castArg4 = arg4.cast<IntegerAttr>();
        auto castArg5 = arg5.cast<IntegerAttr>();
        auto castArg6 = arg6.cast<IntegerAttr>();
        auto castArg7 = arg7.cast<IntegerAttr>();
        auto castArg8 = arg8.cast<IntegerAttr>();

        auto opTypeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, castOpType);
        auto outsizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, castOutsize);
        auto config0Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castConfig0);
        auto config1Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castConfig1);
        auto config2Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castConfig2);
        auto arg0Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg0);
        auto arg1Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg1);
        auto arg2Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg2);
        auto arg3Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg3);
        auto arg4Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg4);
        auto arg5Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg5);
        auto arg6Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg6);
        auto arg7Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg7);
        auto arg8Value = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg8);


        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
        // auto alignmentValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(0));
        auto strAlloca = rewriter.create<LLVM::AllocaOp>(loc, i8PtrType, strLengthValue, 0);

        auto symbolRefPushInst = SymbolRefAttr::get(rewriter.getContext(), "pushInst");
        // Value output = rewriter.create<LLVM::AllocaOp>(loc, f32PtrType, outsizeValue);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefPushInst, ValueRange{//output,
                                                                                        strAlloca,
                                                                                        opTypeValue,
                                                                                       outsizeValue,
                                                                                       config0Value,
                                                                                       config1Value,
                                                                                       config2Value,
                                                                                       arg0Value,
                                                                                       arg1Value,
                                                                                       arg2Value,
                                                                                       arg3Value,
                                                                                       arg4Value,
                                                                                       arg5Value,
                                                                                       arg6Value,
                                                                                       arg7Value,
                                                                                       arg8Value
                                                                                       });
        rewriter.eraseOp(op);
        return success();
    }
};

class ConvertCoreWaitToMLIR : public OpRewritePattern<core::CoreWaitOp> {
public:
    using OpRewritePattern<core::CoreWaitOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreWaitOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto f32Type = rewriter.getF32Type();
        auto i8PtrType = LLVM::LLVMPointerType::get(int8Type);
        auto f32PtrType = LLVM::LLVMPointerType::get(f32Type);

        auto ID = op->getAttr("ID");
        auto castID = ID.cast<StringAttr>();

        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);

        //get id pointer
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
        // auto alignmentValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(0));
        auto strAlloca = rewriter.create<LLVM::AllocaOp>(loc, i8PtrType, strLengthValue, 0);

        for(size_t i=0; i<idStr.size(); i++){
            auto charValue = rewriter.getI8IntegerAttr(idStr[i]);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(i));
            auto charPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{indexValue});
            rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), charValue), charPtr);
        }

        // null terminator
        auto nullTerminatorIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(idStr.size()));
        auto nullTerminatorPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{nullTerminatorIndex});
        rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(0)), nullTerminatorPtr);

        auto symbolRefWaitInst = SymbolRefAttr::get(rewriter.getContext(), "waitInst");
        // Value output = rewriter.create<LLVM::AllocaOp>(loc, f32PtrType, size);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefWaitInst, ValueRange{strAlloca});

        rewriter.eraseOp(op);

        return success();
    }
};

class ConvertCoreReadToMLIR : public OpRewritePattern<core::CoreReadOp> {
public:
    using OpRewritePattern<core::CoreReadOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreReadOp op, PatternRewriter &rewriter) const override{

        auto loc = op.getLoc();

        auto i8PtrType = LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));
        auto int64Type = rewriter.getIntegerType(64);
        auto int32Type = rewriter.getIntegerType(32);
        auto f32Type = rewriter.getF32Type();
        auto f32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());

        // Value size = op.getOperand(0);
        auto sizeAttr = op->getAttr("size").cast<mlir::IntegerAttr>();
        auto sizeValue = rewriter.create<arith::ConstantOp>(loc, int32Type, sizeAttr);

        auto ID = op->getAttr("ID");
        auto arg = op->getAttr("arg");
        auto outShape = op->getAttr("shape");

        auto castID = ID.cast<StringAttr>();
        auto castArg = arg.cast<IntegerAttr>();
        auto castOutShape = outShape.cast<ArrayAttr>();
        
        auto argValue = rewriter.create<arith::ConstantOp>(loc, int32Type, castArg);

        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);


        llvm::SmallVector<int64_t, 1> shape;
        for(auto dim : castOutShape){
            shape.push_back(dim.cast<IntegerAttr>().getInt());
        }

        int64_t totalsize = 1;
        for(int dim : shape){
            totalsize *= dim;
        }

        auto elementType = rewriter.getF32Type();
        auto memrefType = MemRefType::get(shape, elementType);
        
        //get return memory ptr
        Value returnMem = rewriter.create<memref::AllocOp>(loc, memrefType);
        Value returnIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, returnMem);
        Value returnIdxInt = rewriter.create<arith::IndexCastOp>(loc, int64Type, returnIdx);
        Value returnPtr = rewriter.create<LLVM::IntToPtrOp>(loc, f32PtrType, returnIdxInt);
        
        //get id pointer
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
        // auto alignmentValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(0));
        auto strAlloca = rewriter.create<LLVM::AllocaOp>(loc, i8PtrType, strLengthValue, 0);
            //store str to int8* format
        for(size_t i=0; i<idStr.size(); i++){
            auto charValue = rewriter.getI8IntegerAttr(idStr[i]);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(i));
            auto charPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{indexValue});
            rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), charValue), charPtr);
        }

        // null terminator
        auto nullTerminatorIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(idStr.size()));
        auto nullTerminatorPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, strAlloca, ValueRange{nullTerminatorIndex});
        rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(0)), nullTerminatorPtr);


        auto symbolRefReadFromAccel = SymbolRefAttr::get(rewriter.getContext(), "readFromAccel");
// Value output = rewriter.create<LLVM::AllocaOp>(loc, f32PtrType, sizeValue);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefReadFromAccel, ValueRange{//output,
                                                                                           returnPtr,
                                                                                           sizeValue,
                                                                                           strAlloca, 
                                                                                           argValue});
        auto tensorType = RankedTensorType::get(shape, elementType);
        Value returnTensor = rewriter.create<UnrealizedConversionCastOp>(loc, tensorType, returnMem).getResult(0);

        rewriter.create<memref::DeallocOp>(loc, returnMem);

        rewriter.replaceOp(op, returnTensor);

        return success();
    }
};

void CoreToMLIRLoweringPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());

    patterns.add<ConvertCoreAllocToMLIR, ConvertCoreWriteToMLIR, 
        ConvertCoreStartToMLIR, ConvertCoreWaitToMLIR,
        ConvertCoreReadToMLIR>(&getContext());

    OpBuilder builder(module.getContext());
    declareExternalFunction(module, builder);

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
}

std::unique_ptr<Pass> createCoreToMLIRPass() {
    return std::make_unique<CoreToMLIRLoweringPass>();
}

std::unique_ptr<Pass> createCoreToMLIRPass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<CoreToMLIRLoweringPass>(execNodesOnCpu);
}

void declareExternalFunction(mlir::ModuleOp module, mlir::OpBuilder &builder){
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto voidType = LLVM::LLVMVoidType::get(context);
    auto f32Type = FloatType::getF32(context);
    auto int64Type = IntegerType::get(context, 64);
    auto f32PtrType = LLVM::LLVMPointerType::get(f32Type);
    auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);
    auto int32Type = IntegerType::get(context, 32);
    auto charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context,8));


//writeToAccel
    auto writeToAccelFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, int32Type, charPtrType, int32Type, int64PtrType},
        false
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("writeToAccel")){
        auto writeToAccelFunc = builder.create<LLVM::LLVMFuncOp>(loc, "writeToAccel", writeToAccelFuncType);
        module.push_back(writeToAccelFunc);
    }

//readFromAccel
    auto readFromAccelFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{f32PtrType, int32Type, charPtrType, int32Type},
        false
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("readFromAccel")){
        auto readFromAccelFunc = builder.create<LLVM::LLVMFuncOp>(loc, "readFromAccel", readFromAccelFuncType);
        module.push_back(readFromAccelFunc);
    }

//pushInst
    auto pushInstFuncType = LLVM::LLVMFunctionType::get(
        voidType,
        llvm::ArrayRef<mlir::Type>{int32Type,       //optype 
                                   charPtrType,     //instID
                                   int32Type,       //outsize
                                   int32Type,       //config0
                                   int32Type,       //config1
                                   int32Type,       //config2
                                   int32Type,       //arg0
                                   int32Type,       //arg1
                                   int32Type,       //arg2
                                   int32Type,       //arg3
                                   int32Type,       //arg4
                                   int32Type,       //arg5
                                   int32Type,       //arg6
                                   int32Type,       //arg7
                                   int32Type},      //arg8
        false                      
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("pushInst")){
        auto pushInstFunc = builder.create<LLVM::LLVMFuncOp>(loc, "pushInst", pushInstFuncType);
        module.push_back(pushInstFunc);
    }

//waitInst
    auto waitInstFuncType = LLVM::LLVMFunctionType::get(voidType, llvm::ArrayRef<mlir::Type>{charPtrType}, false);
    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("waitInst")){
        auto waitInstFunc = builder.create<LLVM::LLVMFuncOp>(loc, "waitInst", waitInstFuncType);
        module.push_back(waitInstFunc);
    }

}


} // namespace onnx_mlir








