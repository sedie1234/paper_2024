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

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm-c/Core.h"


using namespace mlir;
namespace onnx_mlir {

void declareFunctionCoreOpt(mlir::ModuleOp module, mlir::OpBuilder &builder);

namespace {

#include "src/Accelerators/PA/Conversion/CoreOpt/ONNXCoreOpt.inc"

struct CoreOptPass
    : public PassWrapper<CoreOptPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoreOptPass)

    StringRef getArgument() const override {return "core optimize";}
    StringRef getDescription() const override {
        return "Core Opt Ops";
    }

    CoreOptPass() = default;
    CoreOptPass(const CoreOptPass &pass)
        : PassWrapper<CoreOptPass, OperationPass<ModuleOp>>() {}
    CoreOptPass(mlir::ArrayRef<std::string> execNodesOnCpu){
        this->execNodesOnCpu = execNodesOnCpu;
    }
    void runOnOperation() final;

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnSCpu",
        llvm::cl::desc("Core Optimize"),
        llvm::cl::ZeroOrMore};
};
} // end anonymous namespace

class CoreReadAllocWriteConvert : public OpRewritePattern<core::CoreReadOp> {
public:
    using OpRewritePattern<core::CoreReadOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(core::CoreReadOp op, PatternRewriter &rewriter) const override{

        // if set fused attr, failure.
        if(op->hasAttr("checked")){
            return failure();
        }

        int read_use_count = 0;
        int alloc_use_count = 0;

        Operation* allocOp;
        Operation* writeOp;

        for(auto it : op->getUsers()){
            read_use_count++;
            // it->dump();
            if(it->getName().getStringRef() == "core.Alloc"){
                allocOp = dyn_cast<core::CoreAllocOp>(it);
            }else{
                return failure();
            }
        }

        for(auto it : allocOp->getUsers()){
            alloc_use_count++;
            // it->dump();
            if(it->getName().getStringRef() == "core.Write"){
                writeOp = dyn_cast<core::CoreWriteOp>(it);
            }else{
                return failure();
            }
        }

        if((read_use_count != 1) || (alloc_use_count != 1)){
            return failure();
        }

        // llvm::outs() << read_use_count << " : " << alloc_use_count << "\n";
        // op.dump();
        // allocOp->dump();
        // writeOp->dump();
        // llvm::outs() << "\n\n";

        op->setAttr("checked", rewriter.getUnitAttr());

        auto int64Type = rewriter.getIntegerType(64);
        auto int32Type = rewriter.getIntegerType(32);
        auto int8Type = rewriter.getIntegerType(8);
        auto llvmInt64PtrType = LLVM::LLVMPointerType::get(int64Type);
        auto i8PtrType = LLVM::LLVMPointerType::get(int8Type);
        auto llvmF32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());

        auto loc = op.getLoc();

        auto chain = op->getOperand(2);

        //prev ID set
        auto prevID = op->getAttr("ID");
        auto prevCastID = prevID.cast<StringAttr>();
        std::string prevIdStr = prevCastID.getValue().str();
        auto prevStrLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), prevIdStr.size() + 1);

        //get id pointer
        auto prevStrLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, prevStrLength);
        auto prevStrAlloca = rewriter.create<LLVM::AllocaOp>(loc, i8PtrType, prevStrLengthValue, 0);

        //store str to int8* format
        for(size_t i=0; i<prevIdStr.size(); i++){
            auto charValue = rewriter.getI8IntegerAttr(prevIdStr[i]);
            auto indexValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, rewriter.getI32IntegerAttr(i));
            auto charPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, prevStrAlloca, ValueRange{indexValue});
            rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), charValue), charPtr);
        }

        // null terminator
        auto prevNullTerminatorIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(prevIdStr.size()));
        auto prevNullTerminatorPtr = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, prevStrAlloca, ValueRange{prevNullTerminatorIndex});
        rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(0)), prevNullTerminatorPtr);


        //new ID set
        auto ID = writeOp->getAttr("ID");
        auto castID = ID.cast<StringAttr>();
        std::string idStr = castID.getValue().str();
        auto strLength = rewriter.getIntegerAttr(rewriter.getIntegerType(32), idStr.size() + 1);

        auto argValue = writeOp->getOperand(1);

        //get id pointer
        auto strLengthValue = rewriter.create<LLVM::ConstantOp>(loc, int32Type, strLength);
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


        // get write input shape
        auto shapeAttr = writeOp->getAttr("shape").cast<ArrayAttr>();

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

        for (int i = 0; i < shapeVector.size(); i++) {
            Value idxI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            Value idxI64 = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(i));
            Value _ptr = rewriter.create<LLVM::GEPOp>(loc, llvmInt64PtrType, shapeAlloca, ValueRange{idxI64});
            Value shapeElement = rewriter.create<arith::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(shapeVector[i]));
            rewriter.create<LLVM::StoreOp>(loc, shapeElement, _ptr);
        }

        //externel func call
        auto symbolRefReadAllocWriteOptCallback = SymbolRefAttr::get(rewriter.getContext(), "ReadAllocWriteOptCallback");

        auto result = rewriter.create<LLVM::CallOp>(loc, int32Type, symbolRefReadAllocWriteOptCallback, 
                                                    ValueRange{prevStrAlloca, strAlloca, argValue, chain, shapeAlloca});
        
        rewriter.replaceOp(writeOp, result);

        return success();
    }
};


void CoreOptPass::runOnOperation(){
    auto module = getOperation();
    ConversionTarget target(getContext());

    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());

    OpBuilder builder(module.getContext());
    declareFunctionCoreOpt(module, builder);

    patterns.add<CoreReadAllocWriteConvert>(&getContext());

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    // module.dump();
}

std::unique_ptr<Pass> createCoreOptPass() {
    return std::make_unique<CoreOptPass>();
}

std::unique_ptr<Pass> createCoreOptPass(
    mlir::ArrayRef<std::string> execNodesOnCpu){
    return std::make_unique<CoreOptPass>(execNodesOnCpu);
}

void declareFunctionCoreOpt(mlir::ModuleOp module, mlir::OpBuilder &builder){
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto int64Type = IntegerType::get(context, 64);
    auto int32Type = IntegerType::get(context, 32);
    auto charPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context,8));
    auto int64PtrType = LLVM::LLVMPointerType::get(int64Type);

    auto ReadAllocWriteOptCallbackType = LLVM::LLVMFunctionType::get(
        int32Type,
        llvm::ArrayRef<mlir::Type>{charPtrType, charPtrType, int32Type, int32Type, int64PtrType},
        false
    );

    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("ReadAllocWriteOptCallback")){
        auto ReadAllocWriteOptCallbackFunc = builder.create<LLVM::LLVMFuncOp>(loc, "ReadAllocWriteOptCallback", ReadAllocWriteOptCallbackType);
        module.push_back(ReadAllocWriteOptCallbackFunc);
    }


}

} // namespace onnx_mlir








