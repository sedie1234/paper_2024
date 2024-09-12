MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
rm -rf onnx-mlir/build
mkdir onnx-mlir/build && cd onnx-mlir/build
#cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=${MLIR_DIR} \
        -DONNX_MLIR_ACCELERATORS=PA \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        -DONNX_MLIR_ACCELERATORS=PA \
        ..
fi
cmake --build .
