module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.accels" = ["TA-0x1"], "onnx-mlir.symbol-postfix" = "add"} {
  func.func @main_graph(%arg0: memref<3x2xf32>, %arg1: memref<3x2xf32>) -> memref<3x2xf32> attributes {input_names = ["X1", "X2"], output_names = ["Y"]} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<3x2xf32> to tensor<3x2xf32>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<3x2xf32> to tensor<3x2xf32>
    %2 = "test.Add"(%1, %0) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<3x2xf32> to memref<3x2xf32>
    return %3 : memref<3x2xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 2] , \22name\22 : \22X1\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 2] , \22name\22 : \22X2\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 2] , \22name\22 : \22Y\22 }\0A\0A]\00"} : () -> ()
}
