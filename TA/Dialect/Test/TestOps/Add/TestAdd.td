#ifndef ADD_TD
#define ADD_TD

#ifndef OP_BASE
include "src/Accelerators/TA/Dialect/Test/Test.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

// include "src/Accelerators/TA/Dialect/Test/TestOps/OpHelper.td"

// def TestAdd322Pattern : Pat<
//     (TestAddOp $a, $b, $c),
//     TestAddOp($a, (TestAddOp $b, $c))
// >;

#endif // ADD_TD