#ifndef REFINE_TD
#define REFINE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/IR/AttrBase.td"


//===----------------------------------------------------------------------===//
// Refine Dialect
//===----------------------------------------------------------------------===//

def Refine_Dialect : Dialect {
    let name = "refine";
    let summary = "Refine Dialect";
    let cppNamespace = "::onnx_mlir::refine";
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Attr, operation base class
//===----------------------------------------------------------------------===//

class Refine_Attr<string name, list<Trait> traits = []>
  : BaseLayoutAttr<Refine_Dialect, name, traits>;

class Refine_Op<string mnemonic, list<Trait> traits = []> :
  Op<Refine_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Attribution
//===----------------------------------------------------------------------===//

//TBD


//===----------------------------------------------------------------------===//
// Refine Operations
//===----------------------------------------------------------------------===//

def RefineFusedConvOp : Refine_Op<"FusedConv", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine FusedConv operation";
    let description = [{
        Refine Conv operation
        * input : tensor [N][C][H][W]
        * kernel : tensor [N][C][K][K]
        * bias : tensor [C]
        * kernel_shape : [K]
        * padding : 1D array
        * strides : 1D array
        * output : [N][C][H][W]
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         TensorOf<[F32]>:$kernel,
                         Optional<TensorOf<[F32]>>:$bias,
                         I64ArrayAttr:$kernel_shape,
                         I64ArrayAttr:$padding,
                         I64ArrayAttr:$strides,
                         StrAttr:$ID);

    let results = (outs TensorOf<[F32]>:$output);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::Value":$kernel,
                       "::mlir::Value":$bias,
                       "::mlir::ArrayAttr":$kernel_shape,
                       "::mlir::ArrayAttr":$padding,
                       "::mlir::ArrayAttr":$strides,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineConvOp : Refine_Op<"Conv", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Conv operation";
    let description = [{
        Refine Conv operation
        * input : tensor [N][C][H][W]
        * kernel : tensor [N][C][K][K]
        * bias : tensor [C]
        * kernel_shape : [K]
        * padding : 1D array
        * strides : 1D array
        * output : [N][C][H][W]
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         TensorOf<[F32]>:$kernel,
                         Optional<TensorOf<[F32]>>:$bias,
                         I64ArrayAttr:$kernel_shape,
                         I64ArrayAttr:$padding,
                         I64ArrayAttr:$strides,
                         StrAttr:$ID);

    let results = (outs TensorOf<[F32]>:$output);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::Value":$kernel,
                       "::mlir::Value":$bias,
                       "::mlir::ArrayAttr":$kernel_shape,
                       "::mlir::ArrayAttr":$padding,
                       "::mlir::ArrayAttr":$strides,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineSigmoidOp : Refine_Op<"Sigmoid", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Sigmoid operation";
    let description = [{
        Refine Sigmoid operation (Activation)
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineMulOp : Refine_Op<"Mul", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Mul operation";
    let description = [{
        Refine Mul operation
    }];
    let arguments = (ins TensorOf<[F32]>:$X,
                         TensorOf<[F32]>:$Y,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$X,
                       "::mlir::Value":$Y,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineSplitOp : Refine_Op<"Split", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Split operation";
    let description = [{
        Refine Split operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         TensorOf<[I64]>:$split);
    let results = (outs TensorOf<[F32]>:$output0,
                        TensorOf<[F32]>:$output1);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::Value":$split)>
    ];
    let hasCanonicalizer = 1;
}

def RefineAddOp : Refine_Op<"Add", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Add operation";
    let description = [{
        Refine Add operation
    }];
    let arguments = (ins TensorOf<[F32]>:$X,
                         TensorOf<[F32]>:$Y,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$X,
                       "::mlir::Value":$Y,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineConcatOp : Refine_Op<"Concat", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Concat operation";
    let description = [{
        Refine Concat operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input0,
                         TensorOf<[F32]>:$input1);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input0,
                       "::mlir::Value":$input1)>
    ];
    let hasCanonicalizer = 1;
}

def RefineMaxpoolOp : Refine_Op<"Maxpool", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Refine Maxpool operation";
    let description = [{
        Refine Maxpool operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         I64ArrayAttr:$kernel_shape,
                         I64ArrayAttr:$padding,
                         I64ArrayAttr:$strides,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::ArrayAttr":$kernel_shape,
                       "::mlir::ArrayAttr":$padding,
                       "::mlir::ArrayAttr":$strides,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineResizeOp : Refine_Op<"Resize", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Refine Resize operation";
    let description = [{
        Refine Resize operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input)>
    ];
    let hasCanonicalizer = 1;
}

def RefineTransposeOp : Refine_Op<"Transpose", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Refine Transpose operation";
    let description = [{
        Refine Transpose operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         I64ArrayAttr:$perm);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::ArrayAttr":$perm)>
    ];
    let hasCanonicalizer = 1;
}

def RefineSoftmaxOp : Refine_Op<"Softmax", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Refine Softmax operation";
    let description = [{
        Refine Softmax operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineSliceOp : Refine_Op<"Slice", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Refine Slice operation";
    let description = [{
        Refine Slice operation
    }];
    let arguments = (ins TensorOf<[F32]>:$input,
                         TensorOf<[I64]>:$starts,
                         TensorOf<[I64]>:$ends,
                         TensorOf<[I64]>:$axes,
                         TensorOf<[I64]>:$steps);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::Value":$starts,   
                       "::mlir::Value":$ends,   
                       "::mlir::Value":$axes,   
                       "::mlir::Value":$steps   
                       )>
    ];
    let hasCanonicalizer = 1;
}

def RefineSubOp : Refine_Op<"Sub", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Sub operation";
    let description = [{
        Refine Sub operation
    }];
    let arguments = (ins TensorOf<[F32]>:$X,
                         TensorOf<[F32]>:$Y,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$X,
                       "::mlir::Value":$Y,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}

def RefineDivOp : Refine_Op<"Div", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Refine Div operation";
    let description = [{
        Refine Div operation
    }];
    let arguments = (ins TensorOf<[F32]>:$X,
                         TensorOf<[F32]>:$Y,
                         StrAttr:$ID);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$X,
                       "::mlir::Value":$Y,
                       "::mlir::StringAttr":$ID)>
    ];
    let hasCanonicalizer = 1;
}
#endif // REFINE_TD