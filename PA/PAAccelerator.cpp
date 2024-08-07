#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/PA/Compiler/PACompilerUtils.hpp"
#include "src/Accelerators/PA/Dialect/Refine/RefineOps.hpp"
#include "src/Accelerators/PA/PAAccelerator.hpp"
#include "src/Accelerators/PA/Pass/PAPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include <memory>

#define DEBUG_TYPE "PAAccelerator"

extern llvm::cl::OptionCategory OMPAPassOptions;

namespace onnx_mlir{
namespace accel{

Accelerator *createPA() {return PAAccelerator::getInstance();}
PAAccelerator *PAAccelerator::instance = nullptr;
PAAccelerator *PAAccelerator::getInstance() {
    if(instance == nullptr)
        instance = new PAAccelerator();
    return instance;
}

PAAccelerator::PAAccelerator() : Accelerator(Accelerator::Kind::PA) {
    acceleratorTargets.push_back(this);
};

PAAccelerator::~PAAccelerator() { delete instance; }

uint64_t PAAccelerator::getVersionNumber() const { return 0x00000001; }

void PAAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
    
    addPassesPA(module, pm, emissionTarget, outputNameNoExt);
}

void PAAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
    registry.insert<refine::RefineDialect>();
}

void PAAccelerator::registerPasses(int optLevel) const {
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return onnx_mlir::createONNXToRefinePass();
    });
}


void PAAccelerator::conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const {
    
    target.addLegalDialect<refine::RefineDialect>();
}

mlir::MemRefType PAAccelerator::convertTensorTypeToMemRefType(
        const mlir::TensorType tensorType) const {

    return nullptr;
}

void PAAccelerator::rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
        mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const {
    return;
}

void PAAccelerator::conversionTargetKrnlToLLVM(
        mlir::ConversionTarget &target) const {}

void PAAccelerator::rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
        mlir::LLVMTypeConverter &typeConverter,
        mlir::MLIRContext *ctx) const {
    return;
}

} // namespace accel
} // namespace cool