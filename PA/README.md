# Proto Accelerator on onnx-mlir

## 0. Requirements
1) openssl 설치

```
$ sudo apt install libssl-dev
```


## 1. info
|dialect|operation|간략한 설명|
|:---:|:---:|:---|
|refine|Add|행렬간 덧셈|
|refine|Concat|행렬 병합|
|refine|Conv|convolution|
|refine|Div|두 행렬의 각 element끼리 나눔|
|refine|Maxpool|MAX 방식의 pooling|
|refine|Mul|두 행렬의 각 element끼리 곱셈|
|refine|Resize|행렬의 크기 조정, 복사방식으로 고정|
|refine|Sigmoid|행렬의 각 element에 대해 sigmoid 수행|
|refine|Slice|행렬을 잘라냄|
|refine|Softmax|행렬에 대해 softmax 연산 수행|
|refine|Split|행렬을 2개로 나눔|
|refine|Sub|행렬간 뺄셈|
|refine|Transpose|행렬의 transpose|
|core|Alloc|tensor type을 memref type으로 변경|
|core|Read|accelerator의 메모리로부터 값을 읽음|
|core|Start|명령어 시작|
|core|Wait|지정한 명령어가 수행될 때까지 기다림|
|core|Write|accelerator의 메모리에 값을 씀|


### Refine Dialect
application level에 가까운 기능들을 수행하는 dialect
1) accelerator가 지원하는 operation들 수집. 
2) operation결합, 상쇄, 순서교환

### Core Dialect
accelerator에 가까운 기능들을 수행하는 dialect
1) accelerator를 사용하기 위한 기능들로 구성
2) accelerator용 runtime과 연결
3) 명령어 사이의 상쇄, 순서교환 등 컴파일 기법들 적용

## 2. usage
1. onnx-mlir/src/Accelerator에 복사
2. onnxmlir-build.sh 수정 후 실행
3. onnx-mlir을 원하는 위치에 복사하여 사용 - [예시]()

### build && install
```
$ cp -r PA ./onnx-mlir/src/Accelerator/
$ sh onnxmlir-build-withPA.sh
$ cd onnx-mlir/build/Debug/lib
$ sudo cp libcruntime.a /usr/local/lib/
$ sudo cp PARuntime.a /usr/local/lib/
```

### model compile && inference
1) model compile
```
$ ./onnx-mlir --EmitLib -maccel=PA -lPARuntime -lssl -lcrypto [onnx model]
```

2) inference (python, rough code)
```
from PyRuntime import OMExecutionSession

session = OMExecutionSession(model_shared_lib.so)

output = session.run(input)
```

3) compile options
|category|option|내용|
|:---:|:---:|:---|
|lowering level|--EmitONNXIR|ONNX level까지 lowering|
||--EmitRefineIR|Refine level까지 lowering|
||--EmitCoreHighIR|Core level까지 lowering, core dialect가 포함된 구성|
||--EmitCoreLowIR|Core level까지 lowering, mlir dialect구성까지|
||--EmitMLIR|MLIR level까지 lowering|
||--EmitLLVMIR|LLVM level까지 lowering|
||--EmitLib|shared library(binary) 까지 lowering|
|optimize|--enable-refine-opt|refine level에서 optimize 알고리즘 적용|
||--enable-core-opt|coreHigh level에서 optimize 알고리즘 적용|

