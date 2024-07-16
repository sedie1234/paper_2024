# paper_2024 : Uppering and Blocking on onnx-mlir for Accelerator

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

## 0.2 install
```
$ wget https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7.tar.gz
$ tar -zxvf cmake-3.27.7.tar.gz & cd cmake-3.27.7
$ sudo apt install libssl-dev
$ ./bootstrap
$ make
$ sudo make install
```
```
$ wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protobuf-cpp-3.20.3.tar.gz
$ tar -xzf protobuf-cpp-3.20.3.tar.gz
$ cd protobuf-3.20.3
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
$ sudo ldconfig
$ export Porotobuf_INCLUDE_DIR=/usr/local/include
$ export Protobuf_LIBRARIES=/usr/local/lib/libprotobuf.so
$ export PATH=$PATH:/usr/local/bin
```
```
$ pip install ninja==1.11.1
$ wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
$ unzip ninja-linux.zip
sudo cp ninja /usr/bin
```
```
$ pip insatll torch==2.1.0
$ pip install torchvision
$ sudo apt install python-is-python3 -y
```

## 0.3 Install check
```
$ cmake --version
$ protoc --version
$ pip show ninja
$ pip show torch
$ pip show torchvision
```

## 0.4 build
### 0.4.1 llvm build
```
$ git clone -n https://github.com/llvm/llvm-project.git
$ cd llvm-project && git checkout 91088978d712cd7b33610c59f69d87d5a39e3113 && cd ..

$ sh llvm-build.sh
```

### 0.4.2 onnx-mlir build
```
$ git clone --recursive https://github.com/onnx/onnx-mlir.git
$ cd onnx-mlir
$ git checkout tags/v0.4.1.2
$ git submodule init && git submodule update --recursive
$ cd ..
$ sh onnxmlir-build.sh
$ cd onnx-mlir/build/Debug/lib
$ sudo cp *.so *.a /usr/local/lib/
```

## 0.5 issue
### 0.5.1 memory 부족 이슈
> 메모리가 부족하여 빌드 실패되는 경우<br/>
> 현재 확인된 것은 최대 64 + 40 GB 메모리를 필요로 함<br/>
> 스왑메모리를 할당하여 해결<br/>
#### swap memory allocation
```
$ sudo fallocate -l 50G /swapfile_
$ sudo chmod 600 /swapfile_
$ sudo mkswap /swapfile_
$ sudo swapon /swapfile_
```
#### swap memory check
```
$ sudo swapon --show
$ watch -n 3 swapon --show
```
#### swap memory off
```
$ sudo swapoff /swapfile_
$ sudo rm /swapfile_
```

# 1. Test

## 1.1 add.onnx test(논문에 소개된 내용)
1. add.onnx 모델생성
2. add.onnx.mlir 생성확인
```
$ cd onns-mlir/build/Debug/bin
$ cp onnx-mlir ../../docs/doc-example
$ cd ../../docs/doc-example
$ python3 gen_add_onnx.py
$ ./onnx-mlir --EmitONNXIR add.onnx
```

## 1.2 mnist.onnx test
1. [mnist.onnx](https://github.com/onnx/onnx-mlir/blob/main/docs/mnist_example/mnist.onnx)를 다운
2. onnx-mlir/docs/mnist_example에 복사
```
$ cd onns-mlir/build/Debug/bin
$ cp onnx-mlir ../../../docs/mnist_example
$ cd ../../../docs/mnist_example
$ ./onnx-mlir -O3 -EmitLib mnist.onnx
$ cp ../../build/Debug/lib/*.so .
$ python3 mnist-runPyRuntime.py
```

## 1.3 yolov8n test
1. shared library와 onnx-mlir을 복사해 옴
2. yolov8n.so 파일 생성
3. util/_yolo.py에서 control_session을 수정하여 실행
control_session == 1 이면 onnxruntime으로 실행
control_session == 0 이면 onnx-mlir로 실행
```
$ cp onnx-mlir/build/Debug/lib/*.so myexperiments/yolov8n/
$ cp onnx-mlir/build/Debug/bin/onnx-mlir myexperiments/yolov8n/
$ cd myexperiments/yolov8n
$ ./onnx-mlir --EmitLib --O3 yolov8n.onnx
$ vi util/_yolo.py
contorl_session을 수정
$ python3 image-yolo.py
```

## 1.4 TA (Test Accelerator) test
1. shared library와 onnx-mlir을 복사해 옴
2. 아래 명령어를 실행하여 test.Add가 포함된 mlir 생성
3. EmitONNXIR과 EmitTestIR을 비교
```
$ cp ./onnx-mlir/build/Debug/bin/onnx-mlir* ./myexperiments/TATest/
$ cp ./onnx-mlir/build/Debug/lib/*.so ./myexperiments/TATest
$ cd ./myexperiments/TATest
$ ./onnx-mlir --EmitONNXIR --maccel=TA add.onnx
$ ./onnx-mlir --EmitTestIR --maccel=TA add.onnx
```

# 2. Accelerator 추가

## 2.1 TA : Test Accelerator
1. 0.4.2에서 clone checkout submodule 뒤로 이어짐
2. TA코드를 Accelerator에 추가
3. onnxmlir-build.sh에 Accelerator 설정 추가 : onnxmlir-build-withTA.sh 참고
```
$ cp -r TA onnx-mlir/src/Accelerator/
$ sh onnxmlir-build-withTA.sh
```

