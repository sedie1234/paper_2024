#ifndef TEST_TD
#define TEST_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// Test Dialect
//===----------------------------------------------------------------------===//

def Test_Dialect : Dialect {
  let name = "test";
  let summary = "Test Dialect";
  let cppNamespace = "::onnx_mlir::test";
  let useDefaultAttributePrinterParser = 1;  
}


class Test_Attr<string name, list<Trait> traits = []>
  : BaseLayoutAttr<Test_Dialect, name, traits>;


class Test_Op<string mnemonic, list<Trait> traits = []> :
  Op<Test_Dialect, mnemonic, traits>;

def TestAddOp : Test_Op<"Add", [Pure,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
    let summary = "Test Add operation";
    let description = [{
      Test
    }];
    let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, 
                        TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, 
                        TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
                        
                        AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, 
                        TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, 
                        TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);

    let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, 
                        TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, 
                        TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
    let builders = [
      OpBuilder<(ins "::mlir::Value":$A, "::mlir::Value":$B)>,
    ];
    let hasCanonicalizer = 1;

}

#endif // TEST_TD