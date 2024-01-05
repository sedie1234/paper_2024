# paper_2024 : Uppering and Blocking on onnx-mlir for Acceler

> 1. onnx-mlir에서 지원하는 operation에 대해 acceleration card로 매핑할 수 있게 하기
> 2. uppering이란 과정 만들기
> 3. 성능측정

# 0. Install
> onnx-mlir v0.4.1.2를 기준으로 함

## 0.1 requirements
> cmake (3.27 버전 확인)<br/>
> protobuf (3.20 버전 확인)<br/>
> ninja (pip로 설치, 1.10.2버전 확인, 1.11.1버전 확인)<br/>
> torch (pip로 설치, 2.1.0버전 확인)<br/>
> torchvision 설치 (pip로 설치, 0.16.0버전 확인)<br/>
> numpy 등은 필요에 따라 설치<br/>
> python명령어의 기본 경로를 python3.8(python3.10)으로 바꿔줄 것, python2버전으로 되어있는 경우 오류 발생<br/>

## 0.2 Install check
cmake --version<br/>
protoc --version<br/>
pip show ninja<br/>
pip show torch<br/>
pip show torchvision<br/>

## 0.3 install

### 0.3.1 ninja
1. [ninja releases](https://github.com/ninja-build/ninja/releases)에서 파일을 받아 압축해제
2. sudo cp ninja /usr/bin

## 0.4 build
### 0.4.1 llvm build
  git clone -n https://github.com/llvm/llvm-project.git
  cd llvm-project && git checkout 91088978d712cd7b33610c59f69d87d5a39e3113 && cd ..

  sh llvm-build.sh


### 0.4.2 onnx-mlir build

  git clone --recursive https://github.com/onnx/onnx-mlir.git
  cd onnx-mlir
  git checkout tags/v0.4.1.2
  git submodule init && git submodule update --recursive
  cd ..
  sh onnxmlir-build.sh

