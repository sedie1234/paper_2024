#pragma once

#include "mlir/Pass/Pass.h"

namespace onnx_mlir{

    std::unique_ptr<mlir::Pass> createONNXToRefinePass();
    std::unique_ptr<mlir::Pass> createONNXToRefinePass(
        mlir::ArrayRef<std::string> execNodesOnCpu);

} // namespace onnx_mlir