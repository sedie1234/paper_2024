#pragma once

#include "mlir/Pass/Pass.h"

namespace onnx_mlir{

    std::unique_ptr<mlir::Pass> createONNXToRefinePass();
    std::unique_ptr<mlir::Pass> createONNXToRefinePass(
        mlir::ArrayRef<std::string> execNodesOnCpu);

    std::unique_ptr<mlir::Pass> createRefineToCorePass();
    std::unique_ptr<mlir::Pass> createRefineToCorePass(
        mlir::ArrayRef<std::string> execNodesOnCpu);
        
    std::unique_ptr<mlir::Pass> createCoreToMLIRPass();
    std::unique_ptr<mlir::Pass> createCoreToMLIRPass(
        mlir::ArrayRef<std::string> execNodesOnCpu);

} // namespace onnx_mlir