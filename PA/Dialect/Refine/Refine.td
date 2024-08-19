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
                         TensorOf<[F32]>:$bias,
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

#endif // REFINE_TD