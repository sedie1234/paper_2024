#ifndef REWRITE_ONNX_FOR_TEST
#define REWRITE_ONNX_FOR_TEST

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/TA/Dialect/Test/Test.td"
#endif

class HasRankOf<int rank> :
    Constraint<CPred<"isRankedShapedType($0.getType()) && "
                     "(getRank($0.getType()) == )" # rank # ")">>;

def CreateShapeOp: NativeCodeCall<
    "$_builder.create<mlir::ONNXShapeOp>($_loc, $0, $1, IntegerAttr(), 0)"
>;

def GetShapeTypeOf: NativeCodeCall<
    "RankedTensorType::get({$0.getType().cast<ShapedType>().getRank()}, $_builder.getIntegerType(64))"
>;

def IsUniBroadcastingFromFirstToSecond: Constraint<
    CPred<"isUniBroadcatableFirstToSecond($0, $1)">,
    "Is unidirectional broadcasting from the first to second tensor"
>;

def expandConstantOperandForAddOp1: Pat<
  (ONNXAddOp (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_), $x),
  (ONNXAddOp $x, $c),
  [(IsUniBroadcastingFromFirstToSecond $c, $x)]
>;

def expandConstantOperandForAddOp2: Pat<
  (ONNXAddOp $x, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
  (ONNXAddOp $x, (ONNXExpandOp $c,
                               (CreateShapeOp (GetShapeTypeOf $x), $x),
                               (returnType $x))),
  [(IsUniBroadcastingFromFirstToSecond $c, $x)]
>;

class StringAttrOfValue<string val>:
    NativeCodeCall<"$_builder.getStringAttr(\"" # val # "\")">;

def createArrayAttrOfZerosFrom:
  NativeCodeCall<"createArrayAttrOfZeros($_builder, $0)">;

class FloatAttrOfValue<int val>:
  NativeCodeCall<"createDenseFloatAttrOfValue($_builder, $0, " # val # ")">;

class IsNotStringAttrOfValue<string val>:
  Constraint<CPred<"$0.cast<StringAttr>().getValue() != \"" # val # "\"">>;


#endif // REWRITE_ONNX_FOR_TEST