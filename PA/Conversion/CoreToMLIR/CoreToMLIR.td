#ifndef CORE_TO_MLIR
#define CORE_TO_MLIR

#ifndef OP_BASE

include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/PA/Dialect/Refine/Refine.td"
include "src/Accelerators/PA/Dialect/Core/Core.td"
#endif // OP_BASE

def IsNoneType : Constraint<CPred<"(($_self).getType().isa<NoneType>())">>;
def IsNotNoneType : Constraint<CPred<"(!($_self).getType().isa<NoneType>())">>;

class HasRankOf<int rank> : Constraint<
    CPred<"$0.getType().isa<ShapedType>() && "
          "$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() == " # rank>
>;

def IsBiasNoneOr1D : Constraint<
  CPred<"$_self.getType().isa<NoneType>() || "
        " ($_self.getType().isa<ShapedType>() && "
	"  $_self.getType().cast<ShapedType>().hasRank() && "
	"  $_self.getType().cast<ShapedType>().getRank() == 1)">
>;

class VariadicSizeIs<int N> : Constraint<
  CPred<"$0.size() == " # N>, "The variadic operand's size is N"
>;

class VariadicSizeIsGT<int N> : Constraint<
  CPred<"$0.size() > " # N>, "The variadic operand's size is greater than N"
>;

def CreateNoneValue : NativeCodeCall<"$_builder.create<ONNXNoneOp>($_loc)">;

class getNth<int n, string x> : NativeCodeCall<x # "[" # n # "]">;

class GetNthVariadicOperand<int n> : NativeCodeCall<"$0[" # n # "]">;

class getNthVariadicResults<int n, string x> : NativeCodeCall<x # ".getResults()[" # n # "]">;

def GetShape :
	NativeCodeCall<"$0.getType().cast<ShapedType>().getShape()">;

def GetRank :
	NativeCodeCall<"$0.getType().cast<RankedTensorType>().getRank()">;

def GetSInt :
        NativeCodeCall<"$0.cast<IntegerAttr>().getSInt()">;

class GetStrAttr<string s> :
        NativeCodeCall<"$_builder.getStringAttr(\"" # s # "\")">;

def GetI64Attr :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), $0)">;

class GetI64NAttr<int n> :
	NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), " # n # ")">;

def GetI64ZeroAttr :
        NativeCodeCall<"$_builder.getIntegerAttr($_builder.getIntegerType(64, true), 0)">;

class GetI64ArrayAttr<int n> :
        NativeCodeCall<"$_builder.getI64ArrayAttr(" # n # ")">;

def GetUnrankedTensorTypeOf : NativeCodeCall<
  "UnrankedTensorType::get($0.getType().cast<ShapedType>().getElementType())"
>;

class EmitOp3<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2)">;

class EmitOp4<string s> :
        NativeCodeCall<"$_builder.create<" # s # ">($_loc, $0, $1, $2, $3)">;

// def ReplaceONNXConvPattern : Pat<
//     (ONNXConvOp $input, $kernel, $bias),
//     (RefineConvOp $input, $kernel, $bias, $kernel_shape, $padding, $strides)
// >;

#endif // CORE_TO_MLIR