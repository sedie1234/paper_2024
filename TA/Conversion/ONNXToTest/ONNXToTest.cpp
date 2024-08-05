#define TESTNUM         3


#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
#include "src/Accelerators/TA/Pass/TAPasses.hpp"
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
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "llvm-c/Core.h"

using namespace mlir;

namespace onnx_mlir {

void declareExternalFunction(mlir::ModuleOp module, mlir::OpBuilder &builder);

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

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<bufferization::BufferizationDialect>();
    }

public:
    ListOption<std::string> execNodesOnCpu{*this, "execNodesOnCpu",
        llvm::cl::desc("ONNX To Test Lowering Pass Test"),
        llvm::cl::ZeroOrMore
    };

};
} // end anonymous namespace


class ConvertFuncSignaturePattern : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {

    if(funcOp->hasAttr("signature_converted")){
        std::cout << "signature fail" << std::endl;
        return failure();
    }
        


    std::cout << "signature match and rewrite" << std::endl;

    // 함수의 입력 및 출력 타입을 변환합니다.
    SmallVector<Type, 4> newInputTypes;
    SmallVector<Type, 4> newResultTypes;

    auto convertType = [&](Type type) -> Type {
      if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      }
      return type;
    };

    // 입력 타입 변환
    for (auto argType : funcOp.getFunctionType().getInputs()) {
      newInputTypes.push_back(convertType(argType));
    }

    // 반환 타입 변환
    for (auto resultType : funcOp.getFunctionType().getResults()) {
      newResultTypes.push_back(convertType(resultType));
    }

    // 새로운 함수 타입을 생성합니다.
    auto newFuncType = FunctionType::get(funcOp.getContext(), newInputTypes, newResultTypes);

    // 함수 서명을 업데이트합니다.
    rewriter.updateRootInPlace(funcOp, [&]() {
      funcOp.setType(newFuncType);
      funcOp->setAttr("signature_converted", rewriter.getUnitAttr());
    });

    return success();
  }
};

// class ConvertTensorToMemRefPattern : public OpRewritePattern<Operation> {
// public:
//   using OpRewritePattern<Operation>::OpRewritePattern;

//   LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
//     if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
//       // tensor::ExtractOp을 memref::LoadOp으로 변환
//       Value memref = extractOp.getOperand();
//       Value result = rewriter.create<memref::LoadOp>(extractOp.getLoc(), memref, extractOp.indices());
//       rewriter.replaceOp(extractOp, result);
//       return success();
//     } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
//       // tensor::InsertOp을 memref::StoreOp으로 변환
//       Value memref = insertOp.getOperand(1);
//       rewriter.create<memref::StoreOp>(insertOp.getLoc(), insertOp.getOperand(0), memref, insertOp.indices());
//       rewriter.eraseOp(insertOp);
//       return success();
//     }
//     return failure();
//   }
// };

class ConvertFuncInputsAndOutputs : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {

      // 현재 함수의 입력과 출력 타입을 가져옵니다.

      if(funcOp->hasAttr("converted")){
        std::cout << "FuncInOut fail" << std::endl;
        return failure();
      }
        

      std::cout << "run match and rewrite" << std::endl;      
#if 0

      auto funcType = funcOp.getFunctionType();
      SmallVector<Type, 4> newInputTypes, newResultTypes;

      // 입력 타입 변환 (tensor -> memref)
      for (Type inputType : funcType.getInputs()) {
        if (auto tensorType = inputType.dyn_cast<TensorType>()) {
          newInputTypes.push_back(MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
        } else {
          newInputTypes.push_back(inputType);
        }
      }

      // 출력 타입 변환 (tensor -> memref)
      for (Type resultType : funcType.getResults()) {
        if (auto tensorType = resultType.dyn_cast<TensorType>()) {
          newResultTypes.push_back(MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
        } else {
          newResultTypes.push_back(resultType);
        }
      }

      // 새로운 함수 타입 생성
      auto newFuncType = rewriter.getFunctionType(newInputTypes, newResultTypes);

      // 함수 본문에서 입력과 출력 변환
      for (Block &block : funcOp.getBody()) {
        for (Operation &op : block.getOperations()) {
          if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
            // 반환 타입 변환
            rewriter.setInsertionPoint(returnOp);
            SmallVector<Value, 4> newOperands;
            for (Value operand : returnOp.getOperands()) {
              if (operand.getType().isa<TensorType>()) {
                auto tensorType = operand.getType().cast<TensorType>();
                auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
                auto memRefValue = rewriter.create<bufferization::ToMemrefOp>(returnOp.getLoc(), memRefType, operand);
                newOperands.push_back(memRefValue);
              } else {
                newOperands.push_back(operand);
              }
            }
            rewriter.updateRootInPlace(returnOp, [&] {
              returnOp->setOperands(newOperands);
            });
          }
        }
      }

      // 함수 타입 업데이트
      rewriter.updateRootInPlace(funcOp, [&] {
        funcOp.setType(newFuncType);
        // 변환 완료 속성 추가
        funcOp->setAttr("converted", rewriter.getUnitAttr());
      });

#else

      // 함수 본문에서 입력 및 출력 변환
      IRMapping mapping;
      for (auto &block : funcOp.getBody().getBlocks()) {
        // 블록 인수 변환
        for (auto arg : llvm::enumerate(block.getArguments())) {
          auto tensorType = arg.value().getType().dyn_cast<RankedTensorType>();
          if (tensorType) {
            auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            Value memRefArg = rewriter.create<bufferization::ToMemrefOp>(funcOp.getLoc(), memRefType, arg.value());
            block.getArgument(arg.index()).replaceAllUsesWith(memRefArg);
            mapping.map(arg.value(), memRefArg);
          }
        }
        // 블록 내부 연산 변환
        for (auto &op : block.getOperations()) {
          rewriter.setInsertionPoint(&op);
          SmallVector<Value, 4> newOperands;
          for (auto operand : op.getOperands()) {
            if (auto tensorType = operand.getType().dyn_cast<RankedTensorType>()) {
              auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
              auto memRefValue = rewriter.create<bufferization::ToMemrefOp>(op.getLoc(), memRefType, operand);
              newOperands.push_back(memRefValue);
              mapping.map(operand, memRefValue);
            } else {
              newOperands.push_back(operand);
            }
          }
          rewriter.updateRootInPlace(&op, [&] {
            op.setOperands(newOperands);
          });
        }
      }

      // 변환 완료 속성 추가
      rewriter.updateRootInPlace(funcOp, [&] {
        funcOp->setAttr("converted", rewriter.getUnitAttr());
      });

    std::cout << "FuncInOut End" << std::endl;
#endif

      return success();
    }



  };


class ConvertONNXAddToTest : public OpRewritePattern<ONNXAddOp> {
public:
    using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ONNXAddOp op, PatternRewriter &rewriter) const override {

//===========================================(1)==========================================        
#if TESTNUM==2
        Location loc = op.getLoc();


        // rewriter.setInsertionPointAfter(op);
        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);
        // 새로운 연산을 생성
        auto tensorType = RankedTensorType::get({3, 2}, rewriter.getF32Type());
        Type resultType = op.getResult().getType();
        // Tensor의 값을 정의
        std::vector<float> tensorValues = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto denseAttr = DenseElementsAttr::get<float>(tensorType, tensorValues);

        // // Constant Tensor 생성
        auto constantTensor = rewriter.create<arith::ConstantOp>(loc, tensorType, denseAttr);

        auto testAddOp1 = rewriter.create<onnx_mlir::test::TestAddOp>(loc, resultType, lhs, rhs);
        auto testAddOp2 = rewriter.create<onnx_mlir::test::TestAddOp>(loc, resultType, constantTensor, rhs);
        auto testAddOp3 = rewriter.create<onnx_mlir::test::TestAddOp>(loc, resultType, testAddOp1, rhs);
        
        // 원래 연산을 새로운 연산으로 대체
        rewriter.replaceOp(op, testAddOp1);
        // rewriter.eraseOp(op);
        

        return success();

#endif
//===========================================(2)==========================================
        // Location loc = op.getLoc();

        // Value lhs = op.getOperand(0);
        // Value rhs = op.getOperand(1);
        // Type resultType = op.getResult().getType();


        // // 초기값 설정 (여기서는 tensor 타입으로 설정)
        // // Value initVal = rewriter.create<arith::ConstantOp>(
        // //     loc, resultType, rewriter.getZeroAttr(resultType.cast<TensorType>()));
        // Value initVal = lhs;


        // // scf.while의 조건 정의
        
        // auto condFn = [&](OpBuilder &builder, Location loc, ValueRange args) {
        //     // 첫 번째 값을 로드하여 조건 확인
        //     Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        //     Value firstElem = builder.create<tensor::ExtractOp>(loc, args[0],  ValueRange{zero, zero});
        //     Value cond = builder.create<arith::CmpFOp>(
        //         loc, arith::CmpFPredicate::OGT,
        //         firstElem,
        //         builder.create<arith::ConstantOp>(
        //             loc, rewriter.getF32Type(), rewriter.getFloatAttr(rewriter.getF32Type(), 5.0)));
        //     builder.create<scf::ConditionOp>(loc, cond, args);
        // };

        // // scf.while의 본문 정의
        // auto bodyFn = [&](OpBuilder &builder, Location loc, ValueRange args) {
        //     Value newAdd = builder.create<onnx_mlir::test::TestAddOp>(loc, resultType, lhs, rhs);
        //     builder.create<scf::YieldOp>(loc, newAdd);
        // };

        // // scf.while op 생성
        // rewriter.replaceOpWithNewOp<scf::WhileOp>(op, resultType, initVal, condFn, bodyFn);

//===========================================(3)==========================================

        // Location loc = op.getLoc();

        // // lhs와 rhs를 가져옴
        // // Value lhs = op.getOperand(0);
        // // Value rhs = op.getOperand(1);
        
        // const char *filePath = "./testfile.txt";
        // int fd = open(filePath, O_RDWR);
        // if(fd < 0) {
        //     op.emitError("Failed to open file");
        //     return failure();
        // }

        // void *fileMem = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        // if (fileMem == MAP_FAILED){
        //     close(fd);
        //     op.emitError("Failed to mmap file");
        //     return failure();
        // }

        // // auto memRefType = MemRefType::get({3,2}, rewriter.getF32Type());
        // Value tensorValue = op.getOperand(0);
        // auto tensorType = tensorValue.getType().dyn_cast<TensorType>();
        // auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        // Value srcMemRef = rewriter.create<mlir::UnrealizedConversionCastOp>(
        //     loc, memRefType, tensorValue).getResult(0);

        // Value fixedMemRef = rewriter.create<memref::AllocOp>(loc, memRefType);
        // Value tagMemRef = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1}, rewriter.getI32Type()));

        // // AffineMap 생성
        // AffineMap srcMap = rewriter.getMultiDimIdentityMap(srcMemRef.getType().cast<MemRefType>().getRank());
        // AffineMap dstMap = rewriter.getMultiDimIdentityMap(fixedMemRef.getType().cast<MemRefType>().getRank());
        // AffineMap tagMap = rewriter.getMultiDimIdentityMap(tagMemRef.getType().cast<MemRefType>().getRank());

        // // 전송할 요소의 수 설정
        // Value numElements = rewriter.create<arith::ConstantIndexOp>(loc, 4096);

        // // DMA 시작 (affine.dma_start)
        // rewriter.create<affine::AffineDmaStartOp>(
        //     loc, srcMemRef, srcMap, ValueRange{}, fixedMemRef, dstMap, ValueRange{}, tagMemRef, tagMap, ValueRange{}, numElements);

        // // DMA 완료 대기 (affine.dma_wait)
        // rewriter.create<affine::AffineDmaWaitOp>(loc, tagMemRef, tagMap, ValueRange{}, numElements);

        // rewriter.eraseOp(op);

        // munmap(fileMem, 4096);
        // close(fd);

        // return success();

        // 원래 연산을 새로운 연산으로 대체
        // rewriter.replaceOp(op, add);
   
        // return success();

//===========================================(4)==========================================
        // Location loc = op.getLoc();

        // // 입력 값을 가져옵니다.
        // Value lhs = op.getOperand(0);
        // Value rhs = op.getOperand(1);

        // // lhs와 rhs의 타입을 확인합니다.
        // if (!lhs.getType().isa<RankedTensorType>() || !rhs.getType().isa<RankedTensorType>()) {
        //     return op.emitError("Operands must be of RankedTensorType");
        // }

        // // Tensor 타입을 확인합니다.
        // auto tensorTypeLhs = lhs.getType().cast<RankedTensorType>();
        // auto tensorTypeRhs = rhs.getType().cast<RankedTensorType>();

        // // 결과 타입을 가져옵니다.
        // auto resultType = RankedTensorType::get(tensorTypeLhs.getShape(), tensorTypeLhs.getElementType());
        // // Type resultType = op.getResult().getType();

        // // 새로운 텐서를 생성하기 위한 메모리 할당
        // // auto memRefType = MemRefType::get(resultType.getShape(), resultType.getElementType());
        // // Value alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

        // // SmallVector<Value, 4> dynamicSizes;
        // // for (unsigned i = 0; i < resultType.getRank(); ++i) {
        // //     dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
        // //     std::cout << i << std::endl;
        // // }
        // // std::cout << dynamicSizes.size() << std::endl;
        // // for (const auto &val : dynamicSizes) {
        // //     if (auto op = val.getDefiningOp()) {
        // //         op->dump();  // Operation을 출력
        // //     } else {
        // //         std::cout << "Value is not defined by any Operation." << std::endl;
        // //     }
        // // }

        // Value newTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());

        // // 요소를 더하고 새로운 텐서에 삽입합니다.
        // for (int64_t i = 0; i < tensorTypeLhs.getShape()[0]; ++i) {
        //     for (int64_t j = 0; j < tensorTypeLhs.getShape()[1]; ++j) {
        //         // 인덱스 값 생성
        //         Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
        //         Value jVal = rewriter.create<arith::ConstantIndexOp>(loc, j);

        //         // 요소를 로드하고 AddFOp 생성
        //         Value lhsElem = rewriter.create<tensor::ExtractOp>(loc, lhs, ValueRange{iVal, jVal});
        //         Value rhsElem = rewriter.create<tensor::ExtractOp>(loc, rhs, ValueRange{iVal, jVal});
        //         Value sum = rewriter.create<arith::AddFOp>(loc, lhsElem, rhsElem);


        //         // 결과를 새로운 텐서에 삽입
        //         newTensor = rewriter.create<tensor::InsertOp>(loc, sum, newTensor, ValueRange{iVal, jVal});
        //     }
        // }

        // // 원래 연산을 새로운 연산으로 대체
        // rewriter.replaceOp(op, newTensor);
        // return success();

//===========================================(5)==========================================
        // auto onnxAddOp = cast<mlir::ONNXAddOp>(op);
        // Location loc = op->getLoc();

        // // 1. Tensor를 MemRef로 변환
        // Value tensorValue1 = onnxAddOp.getOperand(0);
        // auto tensorType1 = tensorValue1.getType().dyn_cast<TensorType>();
        // auto memRefType1 = MemRefType::get(tensorType1.getShape(), tensorType1.getElementType());
        // Value srcMemRef1 = rewriter.create<mlir::UnrealizedConversionCastOp>(
        //     loc, memRefType1, tensorValue1).getResult(0);

        // Value tensorValue2 = onnxAddOp.getOperand(1);
        // auto tensorType2 = tensorValue2.getType().dyn_cast<TensorType>();
        // auto memRefType2 = MemRefType::get(tensorType2.getShape(), tensorType2.getElementType());
        // Value srcMemRef2 = rewriter.create<mlir::UnrealizedConversionCastOp>(
        //     loc, memRefType2, tensorValue2).getResult(0);

        // // 2. DMA 쓰기
        // Value fixedMemRef1 = rewriter.create<memref::AllocOp>(loc, memRefType1);
        // Value fixedMemRef2 = rewriter.create<memref::AllocOp>(loc, memRefType2);
        // Value tagMemRef = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1}, rewriter.getI32Type()));

        // AffineMap srcMap = rewriter.getMultiDimIdentityMap(srcMemRef1.getType().cast<MemRefType>().getRank());
        // AffineMap srcMap2 = rewriter.getMultiDimIdentityMap(srcMemRef2.getType().cast<MemRefType>().getRank());
        // AffineMap dstMap1 = rewriter.getMultiDimIdentityMap(fixedMemRef1.getType().cast<MemRefType>().getRank());
        // AffineMap dstMap2 = rewriter.getMultiDimIdentityMap(fixedMemRef2.getType().cast<MemRefType>().getRank());
        // AffineMap tagMap = rewriter.getMultiDimIdentityMap(tagMemRef.getType().cast<MemRefType>().getRank());

        // Value numElements = rewriter.create<arith::ConstantIndexOp>(loc, DMA_DATA_SIZE_);

        // rewriter.create<affine::AffineDmaStartOp>(
        //     loc, srcMemRef1, srcMap, ValueRange{}, fixedMemRef1, dstMap1, ValueRange{}, tagMemRef, tagMap, ValueRange{}, numElements);
        // rewriter.create<affine::AffineDmaStartOp>(
        //     loc, srcMemRef2, srcMap2, ValueRange{}, fixedMemRef2, dstMap2, ValueRange{}, tagMemRef, tagMap, ValueRange{}, numElements);


        // // 3. DMA 기다리기
        // rewriter.create<affine::AffineDmaWaitOp>(loc, tagMemRef, tagMap, ValueRange{}, numElements);

        // // 4. DMA 읽기
        // Value resultMemRef = rewriter.create<memref::AllocOp>(loc, memRefType1);
        // rewriter.create<affine::AffineDmaStartOp>(
        //     loc, fixedMemRef1, dstMap1, ValueRange{}, resultMemRef, dstMap1, ValueRange{}, tagMemRef, tagMap, ValueRange{}, numElements);
        // rewriter.create<affine::AffineDmaStartOp>(
        //     loc, fixedMemRef2, dstMap2, ValueRange{}, resultMemRef, dstMap1, ValueRange{}, tagMemRef, tagMap, ValueRange{}, numElements);


        // // 5. DMA 기다리기
        // rewriter.create<affine::AffineDmaWaitOp>(loc, tagMemRef, tagMap, ValueRange{}, numElements);

        // assert(srcMemRef1.getType().isa<MemRefType>() && "srcMemRef1 Type error");
        // assert(srcMemRef2.getType().isa<MemRefType>() && "srcMemRef2 Type error");
        // assert(fixedMemRef1.getType().isa<MemRefType>() && "fixedMemRef1 Type error");
        // assert(fixedMemRef2.getType().isa<MemRefType>() && "fixedMemRef2 Type error");
        // assert(tagMemRef.getType().isa<MemRefType>() && "tagMemRef Type error");
        // assert(resultMemRef.getType().isa<MemRefType>() && "resultMemRef Type error");

        // // Affine Map 확인
        // assert(srcMap.getNumInputs() == srcMemRef1.getType().cast<MemRefType>().getRank() && "srcMap input size mismatch");
        // assert(dstMap1.getNumInputs() == fixedMemRef1.getType().cast<MemRefType>().getRank() && "dstMap1 input size mismatch");
        // assert(srcMap2.getNumInputs() == srcMemRef2.getType().cast<MemRefType>().getRank() && "srcMap2 input size mismatch");
        // assert(dstMap2.getNumInputs() == fixedMemRef2.getType().cast<MemRefType>().getRank() && "dstMap2 input size mismatch");
        // assert(tagMap.getNumInputs() == tagMemRef.getType().cast<MemRefType>().getRank() && "tagMap input size mismatch");

        // // 6. 결과 값인 MemRef를 다시 Tensor로 변환
        // Value resultTensor = rewriter.create<mlir::UnrealizedConversionCastOp>(
        //     loc, tensorType1, resultMemRef).getResult(0);

        // rewriter.create<memref::DeallocOp>(loc, fixedMemRef1);
        // rewriter.create<memref::DeallocOp>(loc, fixedMemRef2);
        // rewriter.create<memref::DeallocOp>(loc, tagMemRef);
        // rewriter.create<memref::DeallocOp>(loc, resultMemRef);


        // rewriter.replaceOp(op, {resultTensor});
        
        // return success();

//===========================================(6)==========================================

//         Location loc = op.getLoc();
        
//         Value lhs = op.getOperand(0);
//         Value rhs = op.getOperand(1);
//         Type resultType = op->getResult(0).getType();

// llvm::outs() << "Operands retrieved. lhs type: " << lhs.getType() << ", rhs type: " << rhs.getType() << "\n";

//         auto symbolRefWrite = SymbolRefAttr::get(rewriter.getContext(), "file_write");
//         auto symbolRefRead = SymbolRefAttr::get(rewriter.getContext(), "file_read");
//         auto memRefType = MemRefType::get({3, 2}, rewriter.getF32Type());
//         Value lhsMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, memRefType, lhs);
//         Value rhsMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, memRefType, rhs);

// llvm::outs() << "Data copied to memref. lhsMemRef type: " << lhsMemRef.getType() << ", rhsMemRef type: " << rhsMemRef.getType() << "\n";

//         auto symbolRef_write = SymbolRefAttr::get(rewriter.getContext(), "file_write");
//         auto symbolRef_read = SymbolRefAttr::get(rewriter.getContext(), "file_read");

//         // rewriter.create<func::CallOp>(loc, symbolRef_write, ArrayRef<Type>{}, ArrayRef<Value>{lhsMemRef});
//         // rewriter.create<func::CallOp>(loc, symbolRef_write, ArrayRef<Type>{}, ArrayRef<Value>{rhsMemRef});
//         // auto readCallOp = rewriter.create<func::CallOp>(loc, symbolRef_read, ArrayRef<Type>{memRefType}, ArrayRef<Value>{});
//         // Value readResultMemRef = readCallOp.getResult(0);
//         auto readCallOp = rewriter.create<func::CallOp>(loc, TypeRange{LLVM::LLVMPointerType::get(rewriter.getF32Type())}, 
//                                                     symbolRefRead, ValueRange{});
//         Value readResultPtr = readCallOp.getResult(0);
// // llvm::outs() << "Function call results retrieved. readResultMemRef type: " << readResultMemRef.getType() << "\n";

//         Value memRefBuffer = rewriter.create<memref::AllocOp>(loc, memRefType);

//         auto llvmF32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
//         for (int i = 0; i < 3; ++i) {
//             for (int j = 0; j < 2; ++j) {
//                 Value idx[] = {
//                     rewriter.create<arith::ConstantIndexOp>(loc, i),
//                     rewriter.create<arith::ConstantIndexOp>(loc, j)
//                 };
//                 // Compute the pointer to the element
//                 Value elementPtr = rewriter.create<LLVM::GEPOp>(loc, llvmF32PtrType, readResultPtr, idx);
//                 // Load the element
//                 Value element = rewriter.create<LLVM::LoadOp>(loc, elementPtr);
//                 // Store the element in the memref
//                 rewriter.create<memref::StoreOp>(loc, element, memRefBuffer, idx);
//             }
//         }

//         Value readResult = rewriter.create<bufferization::ToTensorOp>(loc, resultType, memRefBuffer);
//         // Value readResult = rewriter.create<bufferization::ToTensorOp>(loc, resultType, readResultMemRef);
// llvm::outs() << "memref to tensor complete. readResult type: " << readResult.getType() << "\n";
        
//         rewriter.replaceOp(op, readResult);
// llvm::outs() << "replaceOp complete.\n";
//         // auto* prevOp = op->getPrevNode();
//         // op->dump();
        
//         // prevOp->dump();
//         // prevOp->getPrevNode()->dump();

//         return success();

//===========================================(7)==========================================
#if TESTNUM==3
llvm::outs() << "Running ConvertONNXAddToTest pass...\n"; // 디버그 출력
        auto loc = op.getLoc();
        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);
        Type resultType = op.getResult().getType();
llvm::outs() << "Operands retrieved.\n"; // 디버그 출력
        // External function references
        auto symbolRefWrite = SymbolRefAttr::get(rewriter.getContext(), "file_write");
        auto symbolRefRead = SymbolRefAttr::get(rewriter.getContext(), "file_read");
llvm::outs() << "External function references created.\n"; // 디버그 출력
        // Convert tensor<3x2xf32> to memref<3x2xf32>
        auto memRefType = MemRefType::get({3, 2}, rewriter.getF32Type());
        // Value lhsMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, memRefType, lhs);
        // Value rhsMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, memRefType, rhs);
        Value lhsMemRef = rewriter.create<UnrealizedConversionCastOp>(loc, memRefType, lhs).getResult(0);
        Value rhsMemRef = rewriter.create<UnrealizedConversionCastOp>(loc, memRefType, rhs).getResult(0);


llvm::outs() << "Data copied to memref.\n"; // 디버그 출력


        // Extract aligned pointer from memref
        auto llvmF32PtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
        auto llvmInt64Type = IntegerType::get(rewriter.getContext(), 64);
        Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, rewriter.getIntegerAttr(llvmInt64Type, 0));
        
        Value lhsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, lhsMemRef);
        Value rhsIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rhsMemRef);

        auto i64Type = rewriter.getIntegerType(64);
        Value lhsIdxInt = rewriter.create<arith::IndexCastOp>(loc, i64Type, lhsIdx);
        Value rhsIdxInt = rewriter.create<arith::IndexCastOp>(loc, i64Type, rhsIdx);

        // Cast the integer pointer to a float pointer
        Value lhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmF32PtrType, lhsIdxInt);
        Value rhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmF32PtrType, rhsIdxInt);


llvm::outs() << "Memref cast to LLVM pointer type.\n"; // 디버그 출력
        // Write lhs and rhs to file
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefWrite, ValueRange{lhsPtr});
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, symbolRefWrite, ValueRange{rhsPtr});
llvm::outs() << "Data written to file.\n"; // 디버그 출력
        // Read result from file
        auto readCallOp = rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmF32PtrType}, symbolRefRead, ValueRange{});
        auto readResultPtr = readCallOp.getResult();

        Value returnMem = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(), memRefType);

        Value returnIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, returnMem);
        Value returnIdxInt = rewriter.create<arith::IndexCastOp>(loc, i64Type, returnIdx);
        Value returnPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmF32PtrType, returnIdxInt);

        Value sizeValue = rewriter.create<arith::ConstantIntOp>(loc, 24, i64Type);

        rewriter.create<LLVM::MemcpyOp>(loc, returnPtr, readResultPtr, sizeValue, false);

        

        // Value returnTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultType, returnMem); 
        Value returnTensor = rewriter.create<UnrealizedConversionCastOp>(loc, resultType, returnMem).getResult(0);  
llvm::outs() << "Read result casted to original type.\n"; // 디버그 출력        
        // Replace the original operation with the result of the read operation
        rewriter.replaceOp(op, returnTensor);
llvm::outs() << "Operation replaced with read result.\n"; // 디버그 출력        




        return success();
#endif
    }
};

void ONNXToTestLoweringPass::runOnOperation() {

#if TESTNUM==1 || TESTNUM==2
    auto module = getOperation();

    std::cout << "onnx to test runOnOperation\r\n";

    onnx_mlir::DimAnalysis dimAnalysis(module);
    dimAnalysis.analyze();

    ConversionTarget target(getContext());

    // target.addLegalDialect<test::TestDialect, memref::MemRefDialect, affine::AffineDialect>();
    // target.addIllegalDialect<ONNXDialect>();
    // target.addIllegalOp<ONNXAddOp>();
    target.addLegalOp<ONNXEntryPointOp>();

    RewritePatternSet patterns(&getContext());
#if TESTNUM==1
    patterns.add<ReplaceONNXAddPattern>(&getContext());
#elif TESTNUM==2
    patterns.add<ConvertONNXAddToTest>(&getContext());
#endif
    
    

    // if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //     signalPassFailure();

    
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));

#endif

//===================================================================================

    // {
    //     ModuleOp module = getOperation();

    //     // 변환 타겟과 패턴을 정의합니다.
    //     ConversionTarget target(getContext());
    //     target.addLegalDialect<affine::AffineDialect, arith::ArithDialect, memref::MemRefDialect>();
    //     target.addIllegalDialect<ONNXDialect>();
    //     target.addLegalOp<ONNXEntryPointOp>();

    //     RewritePatternSet patterns(&getContext());
    //     // patterns.add<ConvertFuncSignatureToMemRef>(&getContext());

    //     // 변환을 적용합니다.
    //     if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //       signalPassFailure();
    //     // (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    // }

//===================================================================================

    // {
    //     ModuleOp module = getOperation();
    //     MLIRContext *context = &getContext();

    //     // 패턴을 등록합니다.
    //     RewritePatternSet patterns(context);
    //     patterns.add<ConvertFuncSignaturePattern>(context);
    //     // patterns.add<ConvertTensorToMemRefPattern>(context);

    //     // 변환 타겟을 정의합니다.
    //     ConversionTarget target(*context);
    //     target.addLegalDialect<func::FuncDialect, memref::MemRefDialect, arith::ArithDialect>();
    //     target.addIllegalOp<tensor::ExtractOp, tensor::InsertOp>();

    //     // 변환을 적용합니다.
    //     if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //     signalPassFailure();
    
    // }

//===================================================================================

    // {
    //     ModuleOp module = getOperation();

    //     // 변환 타겟과 패턴을 정의합니다.
    //     ConversionTarget target(getContext());
    //     target.addLegalDialect<affine::AffineDialect, arith::ArithDialect, memref::MemRefDialect, test::TestDialect,
    //                 tensor::TensorDialect>();
    //     // target.addIllegalDialect<ONNXDialect>();
    //     target.addLegalOp<ONNXEntryPointOp>();


    //     RewritePatternSet patterns(&getContext());
    //     // patterns.add<TensorToMemrefPattern>(&getContext());
    //     patterns.add<ConvertONNXAddToTest>(&getContext());

    //     // 변환을 적용합니다.
    //     if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //       signalPassFailure();
    //     // (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
    // }

//===================================================================================
    // {
    //     ModuleOp module = getOperation();

    //     getContext().loadDialect<bufferization::BufferizationDialect>();

    //     // 변환 패턴 목록 생성
    //     ConversionTarget target(getContext());
    //     RewritePatternSet patterns(&getContext());
    //     patterns.add<ConvertFuncInputsAndOutputs>(&getContext());
    //     patterns.add<ConvertFuncSignaturePattern>(&getContext());
        

    //     // 변환 패턴 적용
    //     if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //         signalPassFailure();
    //     // applyPatternsAndFoldGreedily(module, std::move(patterns));
    // }

//===================================================================================
    
    // {
    //     ModuleOp module = getOperation();
    
    //     RewritePatternSet patterns(&getContext());
    //     patterns.add<ConvertONNXAddToTest>(&getContext());

    //     ConversionTarget target(getContext());
    //     target.addLegalDialect<test::TestDialect, func::FuncDialect>();
    //     target.addIllegalOp<ONNXAddOp>();
    //     target.addLegalOp<ONNXEntryPointOp>();

    //     OpBuilder builder(module.getContext());
    //     declareExternalFunction(module, builder);

    //     if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    //     signalPassFailure();
    //     }        
    //     module->dump();
    // }
//===================================================================================
#if TESTNUM==3
    {
        auto module = getOperation();

        ConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect, test::TestDialect, memref::MemRefDialect>();
        target.addIllegalDialect<ONNXDialect>();
        target.addIllegalOp<ONNXAddOp>();
        target.addLegalOp<ONNXEntryPointOp>();

        RewritePatternSet patterns(&getContext());
        patterns.add<ConvertONNXAddToTest>(&getContext());

        OpBuilder builder(module.getContext());
        declareExternalFunction(module, builder);
        
        // if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        // signalPassFailure();
        // }        
        applyPatternsAndFoldGreedily(module, std::move(patterns));
        // module->dump();
        
    }
#endif
}

std::unique_ptr<Pass> createONNXToTestPass() {
    return std::make_unique<ONNXToTestLoweringPass>();
}

std::unique_ptr<Pass> createONNXToTestPass(
        mlir::ArrayRef<std::string> execNodesOnCpu) {
    return std::make_unique<ONNXToTestLoweringPass>(execNodesOnCpu);
}

// void declareExternalFunction(mlir::ModuleOp module, mlir::OpBuilder &builder){
//     auto context = builder.getContext();
//     auto loc = builder.getUnknownLoc();

//     auto voidType = builder.getNoneType();
//     auto f32PtrType = mlir::RankedTensorType::get({3,2}, builder.getF32Type());
//     auto f32Type = builder.getF32Type();
//     auto memRefType = MemRefType::get({3,2}, f32Type);
//     auto i64Type = builder.getIntegerType(64);

//     auto writeFuncType = builder.getFunctionType(memRefType, voidType);
//     auto fileWriteFunc = mlir::func::FuncOp::create(loc, "file_write", writeFuncType);
//     fileWriteFunc.setPrivate();
//     module.push_back(fileWriteFunc);

//     auto readFuncType = builder.getFunctionType({}, memRefType);
//     auto fileReadFunc = mlir::func::FuncOp::create(loc, "file_read", readFuncType);
//     fileReadFunc.setPrivate();
//     module.push_back(fileReadFunc);

// }


void declareExternalFunction(mlir::ModuleOp module, mlir::OpBuilder &builder){
    auto context = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto voidType = LLVM::LLVMVoidType::get(context);
    auto f32Type = FloatType::getF32(context);
    auto f32PtrType = LLVM::LLVMPointerType::get(f32Type);
    auto writeFuncType = LLVM::LLVMFunctionType::get(voidType, f32PtrType, false);
    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("file_write")){
        auto fileWriteFunc = builder.create<LLVM::LLVMFuncOp>(loc, "file_write", writeFuncType);
        module.push_back(fileWriteFunc);
    }

    auto readFuncType = LLVM::LLVMFunctionType::get(f32PtrType, {}, false);
    if(!module.lookupSymbol<LLVM::LLVMFuncOp>("file_read")){
        auto fileReadFunc = builder.create<LLVM::LLVMFuncOp>(loc, "file_read", readFuncType);
        module.push_back(fileReadFunc);
    }

}

} // namespace onnx_mlir

