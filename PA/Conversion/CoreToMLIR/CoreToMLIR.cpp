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

        auto castID = ID.cast<StringAttr>();

        auto argValue = op.getOperand(1);

    
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

        auto shapeLength = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4);
        //get id pointer
        auto shapeLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, shapeLength);
        // mem alloc
        Value shapeAlloca = rewriter.create<LLVM::AllocaOp>(loc, llvmInt64PtrType, shapeLengthValue, 0);

        // save the shape info
        for (int i = 0; i < shapeVector.size(); i++) {
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, llvmInt64PtrType, shapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(shapeVector[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

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
    
        auto result = rewriter.create<LLVM::CallOp>(loc, int32Type, symbolRefWriteToAccel, 
            ValueRange{ ptr, sizeValue, strAlloca, argValue, shapeAlloca/*shapePtrCast*/});

        rewriter.replaceOp(op, result);



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
        

        auto castID = ID.cast<StringAttr>();


        auto opTypeValue = op->getOperand(0);
        auto outsizeValue = op->getOperand(1);
        auto config0Value = op->getOperand(2);
        auto config1Value = op->getOperand(3);
        auto config2Value = op->getOperand(4);
        auto arg0Value = op->getOperand(5);
        auto arg1Value = op->getOperand(6);
        auto arg2Value = op->getOperand(7);
        auto arg3Value = op->getOperand(8);
        auto arg4Value = op->getOperand(9);
        auto arg5Value = op->getOperand(10);
        auto arg6Value = op->getOperand(11);
        auto arg7Value = op->getOperand(12);
        auto arg8Value = op->getOperand(13);
        auto chainValue = op->getOperand(14);


        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
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


        auto symbolRefPushInst = SymbolRefAttr::get(rewriter.getContext(), "pushInst");
        
        auto result = rewriter.create<LLVM::CallOp>(loc, int32Type, symbolRefPushInst, ValueRange{//output,
                                                                                        opTypeValue,
                                                                                        strAlloca,
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
                                                                                       arg8Value,
                                                                                       chainValue
                                                                                       });
        
        rewriter.replaceOp(op, result);

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


        auto chain = op->getOperand(0);

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
        
        auto result = rewriter.create<LLVM::CallOp>(loc, int32Type, symbolRefWaitInst, ValueRange{strAlloca, chain});
        
        
        rewriter.replaceOp(op, result);



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

        auto sizeValue = op->getOperand(0);
        auto argValue = op->getOperand(1);
        auto chain = op->getOperand(2);

        auto ID = op->getAttr("ID");
        auto outShape = op->getAttr("shape");


        auto castID = ID.cast<StringAttr>();
        auto castOutShape = outShape.cast<ArrayAttr>();
    

        std::string idStr = castID.getValue().str();

        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);

        auto result = op.getResult();
        auto resultType = result.getType().dyn_cast<mlir::RankedTensorType>();
        
        llvm::ArrayRef<int64_t> result_shape = resultType.getShape();

        int64_t totalsize = 1;
        for(int dim : result_shape){
            totalsize *= dim;
        }
        

        // llvm::outs() << "CoreReadOp : " << castID << "\n";
        // llvm::outs() << "tatalsize : sizeValue = ";
        // for(int dim : result_shape){
        //     llvm::outs() << dim << ", ";
        // }
        // llvm::outs() << totalsize << " \n";
        // sizeValue.dump();
        // llvm::outs() << " \n";

        auto elementType = rewriter.getF32Type();
        // auto memrefType = MemRefType::get(shape, elementType);
        auto memrefType = MemRefType::get(result_shape, elementType);
        
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

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefReadFromAccel, ValueRange{//output,
                                                                                           returnPtr,
                                                                                           sizeValue,
                                                                                           strAlloca, 
                                                                                           argValue,
                                                                                           chain});

        Value returnTensor = rewriter.create<UnrealizedConversionCastOp>(loc, resultType, returnMem).getResult(0);

        // rewriter.create<memref::DeallocOp>(loc, returnMem);

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
        int32Type,
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
        llvm::ArrayRef<mlir::Type>{f32PtrType, int32Type, charPtrType, int32Type, int32Type},
        false
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("readFromAccel")){
        auto readFromAccelFunc = builder.create<LLVM::LLVMFuncOp>(loc, "readFromAccel", readFromAccelFuncType);
        module.push_back(readFromAccelFunc);
    }

//pushInst
    auto pushInstFuncType = LLVM::LLVMFunctionType::get(
        int32Type,
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
                                   int32Type,       //arg8
                                   int32Type},      //chain
        false                      
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("pushInst")){
        auto pushInstFunc = builder.create<LLVM::LLVMFuncOp>(loc, "pushInst", pushInstFuncType);
        module.push_back(pushInstFunc);
    }

//waitInst
    auto waitInstFuncType = LLVM::LLVMFunctionType::get(int32Type, llvm::ArrayRef<mlir::Type>{charPtrType, int32Type}, false);
    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("waitInst")){
        auto waitInstFunc = builder.create<LLVM::LLVMFuncOp>(loc, "waitInst", waitInstFuncType);
        module.push_back(waitInstFunc);
    }

}


} // namespace onnx_mlir








