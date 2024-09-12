#include <math.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Accelerators/PA/Dialect/Core/CoreOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace core {

void CoreDialect::initialize(){
    addOperations<
    #define GET_OP_LIST
    #include "src/Accelerators/PA/Dialect/Core/CoreOps.cpp.inc"
    >();
}


} // namespace core
} // namespace onnx_mlir


#define GET_OP_CLASSES
#include "src/Accelerators/PA/Dialect/Core/CoreOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/PA/Dialect/Core/CoreAttributes.cpp.inc"

#include "src/Accelerators/PA/Dialect/Core/CoreDialect.cpp.inc"