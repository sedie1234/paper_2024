# yolov8n 기반의 onnx-mlir 동작실험

## 0. 사전준비
### 0.1. PA를 포함한 onnx-mlir 빌드
1) [PA를 포함한 onnx-mlir 빌드](https://github.com/sedie1234/paper_2024/blob/main/PA/README.md)
2) onnx-mlir 복사해오기
```
$ sh onnxmlir-build-withPA.sh
$ sudo cp ./onnx-mlir/build/Debug/bin/onnx-mlir* ./myexperiments/yolov8n/
$ sudo cp ./onnx-mlir/build/Debug/lib/*.so ./myexperiments/yolov8n/
$ sudo cp ./onnx-mlir/build/Debug/lib/libcruntime.a /usr/local/lib/
$ sudo cp ./onnx-mlir/build/Debug/lib/libPARuntime.a /usr/local/lib/
```

### 0.2 myaccel 빌드 및 실행
[myaccel 빌드 및 실행](https://github.com/sedie1234/paper_2024/tree/main/myexperiments/myAccelProto)

## 1. 실험 개요
### 1.1 사용할 모델 제작
기본적으로 yolov8n이 포함되어 있음. 

1) yolov8n 기반의 모델 제작
이 모델을 이용해서 모델의 일부분만 테스트할 수 있는 모델을 생성하는 코드 : yolo_split.py
8,9번줄에 생성할 모델파일의 이름 filename1, filename2를 수정, 
10번 줄에 모델을 나누는 기준이 되는 tensor 이름 설정 후 실행
```
$ python3 yolo_split.py
```

2) yolov8n의 특정 노드만 존재하는 모델 제작
```
$ python3 makeconv.py
$ python3 makemul.py
$ python3 makesig.py
```


### 1.2 모델 컴파일
모델을 컴파일하여 shared library 생성
```
$ ./onnx-mlir --EmitLib -maccel=PA -lPARuntime -lssl -lcrypto mymodel.onnx
```

### 1.3 inference code 작성
1.2에서 생성한 shared library를 이용하여 session을 열고 inference 수행
```
import numpy as np
from PyRuntime import OMExecutionSession

session = OMExecutionSession(model file name)

intput = np.random.rand(1, 16, 320, 320).astype(np.float32)
output = session.run(input)
```

## 2. 실험별 상세내용
0번 사전준비가 모두 수행된 후 진행
### 2.1 yolov8n.onnx
 - **메모리 문제로 끝까지 동작하지는 않음.** memref::AllocOp가 제대로 메모리를 할당하지 못하는 문제로 추정 중
 - /util/_yolo.py/ 의 14~17번줄 아래와 같이 수정 후 실행
```
control_session = 0 # 2: compare onnxruntime & onnx-mlir // 1: onnxruntime // 0: onnx-mlir
model_split = 0
ort_on = 0
om_on = 0
```

```
$ ./onnx-mlir --EmitLib -maccel=PA -lPARuntime -lssl -lcrypto yolov8n.onnx
$ ../myAccelProto/run/myaccel (다른 터미널에서 실행)
$ python3 image_yolo.py
```

### 2.2 sigmoid, mul, conv 실험
 - sigmoid만 있는 모델을 생성하여 sigmoid가 정상적으로 동작하는지 실험
 - makesig.py와 run_sigmoid_test.py에서 input shape가 동일해야 함
 - sig 또는 sigmoid를 mul, conv로 바꾸면 동일하게 실험진행 가능
 - 모두 정상적으로 동작하는 것 확인
```
$ python3 makesig.py
$ ./onnx-mlir --EmitLib -maccel=PA -lPARuntime -lssl -lcrypto sigmoid_test_model.onnx
$ ../myAccelProto/run/myaccel (다른 터미널에서 실행)
$ python3 run_sigmoid_test.py
```

### 2.3 분할된 yolov8n 실험
 - yolov8n의 일부분만으로 구성된 모델을 제작하여 실험
 - yolo_split.py의 split_tensor를 수정하여 노드를 1개만 포함 / 9개만 포함하는 방식으로 동작 확인
 - 동작은 정상적으로 이루어지지만 **노드를 지날 때마다 값의 loss가 발생**하여 9개의 노드를 지난 뒤에는 평균적으로 0.1정도의 값이 차이가 발생함
 - 노드를 1개만 포함하는 방식
   split_tensor = "/model.0/conv/Conv_output_0"
 - 노드를 9개만 포함하는 방식
   split_tensor = "/model.2/cv1/act/Mul_output_0"
 - util/_yolo.py의 14~18번 줄과 44, 45번 줄 수정
 - 방법 1 : control_session=2, model_split=0으로 하고 ort_on과 om_on을 0과 1로 바꿔가며 실험
 - 방법 2 : control_session=1로 두어 onnxruntime으로 inference 수행 (yolo모델이어야 함)
 - 방법 3 : control_session=0으로 두어 onnx-mlir로 inference 수행 (yolo 모델이어야 함, 현재는 안됨)
```
14 control_session = 2 # 2: compare onnxruntime & onnx-mlir // 1: onnxruntime // 0: onnx-mlir
15 model_split = 0
16 ort_on = 1
17 om_on = 1
18 model_base = './part9'

44 self.session = onnxruntime.InferenceSession('./_9_part1.onnx')
45 self.sessions.append(OMExecutionSession('./_9_part1.so'))
```

```
$ python3 yolo_split.py
$ ./onnx-mlir --EmitLib -maccel=PA -lPARuntime -lssl -lcrypto _9_part.onnx
$ python3 image_yolo.py --app test
$ python3 compare.py
```

## Error log
### 1. valgrind - memory 할당에서 segmentation fault
 - valgrind 실행 결과 PARuntime.c의 fread(data, 1, datasize, file); 에서 할당되지 않은 주소에 접근하는 것으로 나옴
 - data는 memref::AllocOp를 통해 메모리를 할당받고 그 주소를 넘겨받아, 0x128811040 주소를 가르킴
 - 에러는 0x128811FFF 주소에 접근하는 것으로 segmentation fault가 발생함
   
### 2. _9_part1 실험
 - _9_part1 실험에서 onnx-mlir만 돌릴 때는 정상동작, onnxruntime session run까지 포함하면 onnx-mlir에서 에러발생
 - 프로그램이 사용하는 메모리와 관련이 있는 것으로 예상
