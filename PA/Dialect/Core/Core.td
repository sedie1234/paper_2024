#ifndef CORE_TD
#define CORE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/IR/AttrBase.td"


//===----------------------------------------------------------------------===//
// Core Dialect
//===----------------------------------------------------------------------===//

def Core_Dialect : Dialect {
    let name = "core";
    let summary = "Core Dialect";
    let cppNamespace = "::onnx_mlir::core";
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Attr, operation base class
//===----------------------------------------------------------------------===//

class Core_Attr<string name, list<Trait> traits = []>
    : BaseLayoutAttr<Core_Dialect, name, traits>;

class Core_Op<string mnemonic, list<Trait> traits = []> 
    : Op<Core_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Attribution
//===----------------------------------------------------------------------===//

//TBD


//===----------------------------------------------------------------------===//
// Core Operations
//===----------------------------------------------------------------------===//

def CoreAllocOp : Core_Op<"Alloc", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Core Alloc operation";
    let description = [{
        Core Alloc operation
    }];
    let arguments = (ins TensorOf<[F32, I64]>:$input);
    let results = (outs MemRefOf<[F32, I64]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input)>
    ];
    let hasCanonicalizer = 1;
}

def CoreWriteOp : Core_Op<"Write", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Core Write operation";
    let description = [{
        Core Write opertaion
    }];
    let arguments = (ins MemRefOf<[F32]>:$input,
                         StrAttr:$ID,
                         I32:$arg,
                         ArrayAttr:$shape);
    let results = (outs I32:$addr);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,
                       "::mlir::StringAttr":$ID,
                       "::mlir::IntegerAttr":$arg,
                       "::mlir::ArrayAttr":$shape)>
    ];
    let hasCanonicalizer = 1;
}

def CoreStartOp : Core_Op<"Start", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Core Start operation";
    let description = [{
        CoreStart operation
    }];
    let arguments = (ins
                      StrAttr:$ID,
                      I32:$optype,
                      I32:$outsize,
                      I32:$config0,
                      I32:$config1,
                      I32:$config2,
                      I32:$arg0,
                      I32:$arg1,
                      I32:$arg2,
                      I32:$arg3,
                      I32:$arg4,
                      I32:$arg5,
                      I32:$arg6,
                      I32:$arg7,
                      I32:$arg8,
                      I32:$chain
                      );
    let results = (outs I32:$out);
    let builders = [
        OpBuilder<(ins "::mlir::StringAttr":$ID,
                       "::mlir::IntegerAttr":$optype,
                       "::mlir::IntegerAttr":$outsize,
                       "::mlir::IntegerAttr":$config0,
                       "::mlir::IntegerAttr":$config1,
                       "::mlir::IntegerAttr":$config2,
                       "::mlir::IntegerAttr":$arg0,
                       "::mlir::IntegerAttr":$arg1,
                       "::mlir::IntegerAttr":$arg2,
                       "::mlir::IntegerAttr":$arg3,
                       "::mlir::IntegerAttr":$arg4,
                       "::mlir::IntegerAttr":$arg5,
                       "::mlir::IntegerAttr":$arg6,
                       "::mlir::IntegerAttr":$arg7,
                       "::mlir::IntegerAttr":$arg8,
                       "::mlir::IntegerAttr":$chain)>
    ];
    let hasCanonicalizer = 1;
}

def CoreWaitOp : Core_Op<"Wait", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

    let summary = "Core Wait operation";
    let description = [{
        CoreWait operation
    }];
    let arguments = (ins StrAttr:$ID,
                         I32:$chain);
    let results = (outs I32:$out);
    let builders = [
        OpBuilder<(ins "::mlir::StringAttr":$ID,
                       "::mlir::IntegerAttr":$chain)>
    ];
    let hasCanonicalizer = 1;
}

def CoreReadOp : Core_Op<"Read", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]>{

    let summary = "Core Read operation";
    let description = [{
        CoreRead operation
    }];
    let arguments = (ins I32:$size,
                         StrAttr:$ID,
                         I32:$arg,
                         ArrayAttr:$shape,
                         I32:$chain);
    let results = (outs TensorOf<[F32]>:$output);
    let builders = [
        OpBuilder<(ins "::mlir::IntegerAttr":$size,
                       "::mlir::StringAttr":$ID,
                       "::mlir::IntegerAttr":$arg,
                       "::mlir::ArrayAttr":$shape,
                       "::mlir::IntegerAttr":$chain)>
    ];
    let hasCanonicalizer = 1;
}


#endif // CORE_TD
