// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Fuse both Sqrt to Add

func.func @test_fuse_element3(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    return %2 : tensor<1024xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_fuse_element3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[VAR_4_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.sqrt [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024xf32>
// CHECK:         }
}

// -----

// Stop fusion after the first Sqrt because it has more one user

func.func @test_fuse_element4(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %3 : tensor<1024xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_fuse_element4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = math.sqrt [[VAR_6_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_1_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_6_1_]], [[RES_2_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<1024xf32>
// CHECK:         }
}

// -----

// Start from unary op and dynamic

func.func @test_fuse_element7(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) -> tensor<?xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_fuse_element7
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[VAR_4_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
}

// -----

// Start from binary and last fusible Op has different element type

func.func @test_fuse_element8(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) -> tensor<?xi8> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<?xf32>) -> tensor<?xi8>
  return %1 : tensor<?xi8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_fuse_element8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?xi8> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.fptosi [[VAR_4_]] : f32 to i8
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xi8>
// CHECK:         }
}

// -----

// fusible for binary with block argument input
func.func @fuse_element_10(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:  func.func @fuse_element_10
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xf32>, [[PARAM_1_:%.+]]: memref<4x5xf32>) -> memref<4x5xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xf32>
// CHECK:         }
}

// -----

// fusible binary with constant input
func.func @fuse_element_14(%arg0: tensor<5xf32>) -> tensor<*xf32> {
  %cst = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  %0 = "onnx.Sqrt"(%arg0) : (tensor<5xf32>) -> tensor<?xf32>
  %1 = "onnx.Add"(%0, %cst) : (tensor<?xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:  func.func @fuse_element_14
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5xf32>) -> memref<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [5], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> memref<5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_4_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5xf32>
// CHECK:         }
}
