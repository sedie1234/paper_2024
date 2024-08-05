#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
// #include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"

#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/TA/Compiler/TACompilerOptions.hpp"
#include "src/Accelerators/TA/Compiler/TACompilerUtils.hpp"
#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
#include "src/Accelerators/TA/Pass/TAPasses.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "TACompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void addONNXToTestPasses(mlir::PassManager &pm, ArrayRef<std::string> execNodesOnCpu){

    for (unsigned i=0; i<3; i++){
        pm.addPass(onnx_mlir::createRewriteONNXForTestPass(execNodesOnCpu));
        pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
    }

    unsigned instrumentActions = instrumentControlBits;
    if (profileIR == onnx_mlir::ProfileIRs::Test){
        instrumentStage = onnx_mlir::InstrumentStages::Test;
        instrumentOps = "onnx.*,test.*";
        instrumentActions |= (1 << 3) - 1;
    }

    if (instrumentStage == onnx_mlir::InstrumentStages::Onnx)
        pm.addNestedPass<func::FuncOp>(
            onnx_mlir::createInstrumentPass(instrumentOps, instrumentActions));
        
    pm.addPass(onnx_mlir::createONNXToTestPass(execNodesOnCpu));
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    pm.addPass(mlir::createCanonicalizerPass());  

    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(createScrubDisposablePass());

    pm.addPass(mlir::createCSEPass());

    // pm.addPass(mlir::createConvertArithToLLVMPass());


}

void addPassesTA(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt){

    // LLVM_DEBUG(llvm::dbgs() << "Adding TA passes" << std::endl;);
    if (emissionTarget >= EmitONNXIR)
        addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());

    if (emissionTarget >= EmitMLIR) {
        addONNXToTestPasses(pm, execNodesOnCpu);

        if(taEmissionTarget >= EmitTestIR)
            emissionTarget = EmitMLIR;
        else{
            pm.addPass(mlir::createCanonicalizerPass());

            std::string optStr = getCompilerOption(OptionKind::CompilerOptLevel);
            OptLevel optLevel = OptLevel::O0;
            if (optStr == "-O0")
                optLevel = OptLevel::O0;
            else if (optStr == "-O1")
                optLevel = OptLevel::O1;
            else if (optStr == "-O2")
                optLevel = OptLevel::O2;
            else if (optStr == "-O3")
                optLevel = OptLevel::O3;
            addONNXToKrnlPasses(pm, optLevel, true, instrumentONNXSignature, ONNXOpStats);

        }
    }

    if (emissionTarget >= EmitLLVMIR)
        addKrnlToLLVMPasses(pm, outputNameNoExt, true);

}

} // namespace onnx_mlir




