#include "src/Accelerators/TA/Compiler/TACompilerOptions.hpp"

#define DEBUG_TYPE "TACompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<TAEmissionTargetType> taEmissionTarget(
    llvm::cl::desc("test... taEmissionTarget description."),
    llvm::cl::values(
        clEnumVal(EmitTestIR, "Test IR"),
        clEnumVal(EmitTestNone, "Do not emit Test IR")
    ),
    llvm::cl::init(EmitTestNone), llvm::cl::cat(OnnxMlirOptions)
);

llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
    llvm::cl::desc("execNodesOnCpu option test description..."),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
    llvm::cl::cat(OnnxMlirOptions)
};

llvm::cl::opt<bool> taEnableTestToOnnx("enable-test-to-onnx",
    llvm::cl::desc("test taEnableTestToOnnx description test..."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions)
);

} // namespace onnx_mlir