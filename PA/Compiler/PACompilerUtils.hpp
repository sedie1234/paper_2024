#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

void addONNXToRefinePasses(mlir::PassManager &pm/*, ArrayRef<std::string> execNodesOnCpu*/);
void addRefineToCorePasses(mlir::PassManager &pm/*, ArrayRef<std::string> execNodesOnCpu*/);


void addPassesPA(mlir::OwningOpRef<mlir::ModuleOp> &module, 
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt);

} // namespace onnx_mlir