#include "src/Accelerators/PA/Compiler/PACompilerOptions.hpp"

#define DEBUG_TYPE "PACompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<PAEmissionTargetType> paEmissionTarget(
    llvm::cl::desc("PA is for the prototype accelerator."),
    llvm::cl::values(
        clEnumVal(EmitRefineIR, "Lower model to Refine IR"),
        clEnumVal(EmitCoreIR, "Lower model to Core IR"),
        clEnumVal(EmitPNONE, "Do not emit any PA IR")
    ),
    llvm::cl::init(EmitPNONE), llvm::cl::cat(OnnxMlirOptions)
);

llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
    llvm::cl::desc("execNodesOnCpu"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
    llvm::cl::cat(OnnxMlirOptions)
};

llvm::cl::opt<bool> paEnableRefineToOnnx("enable-refine-to-onnx",
    llvm::cl::desc("paEnableRefineToOnnx"),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions)
);

} // namespace onnx_mlir