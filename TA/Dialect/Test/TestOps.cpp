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

#include "src/Accelerators/TA/Dialect/Test/TestOps.hpp"
// #include "src/Accelerators/TA/Dialect/Test/TestOps/OpHelper.hpp"
// #include "src/Accelerators/TA/Dialect/Test/TestOps/ShapeHelper.hpp"
// #include "src/Accelerators/TA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/Diagnostic.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;


namespace onnx_mlir {
namespace test {

void TestDialect::initialize() {
//   addAttributes<
// #define GET_ATTRDEF_LIST
// #include "src/Accelerators/TA/Dialect/Test/TestAttributes.cpp.inc"
//       >();
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/TA/Dialect/Test/TestOps.cpp.inc"
      >();
}

} // namespace test
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
// Keep this part at the end of the file.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/TA/Dialect/Test/TestOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
// #include "src/Accelerators/TA/Dialect/Test/TestAttributes.cpp.inc"

#include "src/Accelerators/TA/Dialect/Test/TestDialect.cpp.inc"
