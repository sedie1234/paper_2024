#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/TA/Compiler/TACompilerUtils.hpp"
#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
#include "src/Accelerators/TA/TAAccelerator.hpp"
#include "src/Accelerators/TA/Pass/TAPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <memory>

#define DEBUG_TYPE "TAAccelerator"

extern llvm::cl::OptionCategory OMTAPassOptions;

namespace onnx_mlir{
namespace accel{

Accelerator *createTA() { return TAAccelerator::getInstance(); }
TAAccelerator *TAAccelerator::instance = nullptr;
TAAccelerator *TAAccelerator::getInstance() {
    if (instance == nullptr)
        instance = new TAAccelerator();
    return instance;
}

TAAccelerator::TAAccelerator() : Accelerator(Accelerator::Kind::TA) {
    LLVM_DEBUG(llvm::dbgs() << "Creating an TA accelerator\n");
    acceleratorTargets.push_back(this);
    // addCompilerConfig(CCM_SHARED_LIB_DEPS, {});
};

TAAccelerator::~TAAccelerator() { delete instance; }

uint64_t TAAccelerator::getVersionNumber() const { return 0x00000001; }

void TAAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
        mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
        std::string outputNameNoExt) const {
    LLVM_DEBUG(llvm::dbgs() << "Adding passes for TA accelerator\n");
    addPassesTA(module, pm, emissionTarget, outputNameNoExt);
}

void TAAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
    LLVM_DEBUG(llvm::dbgs() << "Registering dialects for TA accelerator\n");
    registry.insert<test::TestDialect>();
}

void TAAccelerator::registerPasses(int optLevel) const {
    LLVM_DEBUG(llvm::dbgs() << "Registering passes for TA accelerator\n");
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return onnx_mlir::createONNXToTestPass();
    });

    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return onnx_mlir::createRewriteONNXForTestPass();
    });
}

mlir::MemRefType TAAccelerator::convertTensorTypeToMemRefType(
        const mlir::TensorType tensorType) const {

    return nullptr;
}

void TAAccelerator::conversionTargetONNXToKrnl(
        mlir::ConversionTarget &target) const {
    target.addLegalDialect<test::TestDialect>();
}

void TAAccelerator::rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
        mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const {
    return;
}

void TAAccelerator::conversionTargetKrnlToLLVM(
        mlir::ConversionTarget &target) const {}

void TAAccelerator::rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
        mlir::LLVMTypeConverter &typeConverter,
        mlir::MLIRContext *ctx) const {
    return;
}

} // namespace accel
} // namespace onnx_mlir

