#pragma once

#include "mlir/Pass/Pass.h"

namespace onnx_mlir{

    std::unique_ptr<mlir::Pass> createONNXToTestPass();
    std::unique_ptr<mlir::Pass> createONNXToTestPass(
        mlir::ArrayRef<std::string> execNodesOnCpu);

    std::unique_ptr<mlir::Pass> createRewriteONNXForTestPass();
    std::unique_ptr<mlir::Pass> createRewriteONNXForTestPass(
        mlir::ArrayRef<std::string> execNodesOnCpu);

} // namespace onnx_mlir