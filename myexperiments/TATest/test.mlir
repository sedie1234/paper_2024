module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.accels" = ["TA-0x1"], "onnx-mlir.symbol-postfix" = "add"} {
  func.func @main_graph(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {input_names = ["X1", "X2"], output_names = ["Y"]} {
    %0 = "test.Add"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
