# paper_2024 : Uppering and Blocking on onnx-mlir for Acceler

> 1. onnx-mlir에서 지원하는 operation에 대해 acceleration card로 매핑할 수 있게 하기
> 2. uppering이란 과정 만들기
> 3. 성능측정

# 0. Install
> onnx-mlir v0.4.1.2를 기준으로 함

## 0.1 requirements
> cmake (3.27 버전 확인)
> protobuf (3.20 버전 확인)
> ninja (pip로 설치, 1.10버전 확인)
> torch (pip로 설치, 2.1.0버전 확인)
> torchvision 설치 (pip로 설치, 0.16.0버전 확인)
> numpy 등은 필요에 따라 설치
> python명령어의 기본 경로를 python3.8(python3.10)으로 바꿔줄 것, python2버전으로 되어있는 경우 오류 발생

## 0.2 Install check
cmake --version
protoc --version
pip show ninja
pip show torch
pip show torchvision

## 0.3 install

### 0.3.1 ninja
pip install ninja

