# myAccel
- PA와 연동되도록 설계한 myaccelerator, 하드웨어가 없는 환경에서 가상의 하드웨어 역할을 수행.
- memory/ 디렉토리에 0~299까지 파일을 만들어 accelerator의 bank역할
- shared memory로 레지스터 역할 수행
- interrupt를 통해 동작제어

------------[그림]------------

## myaccel build
```
$ cd src
$ mkdir build & cd build
$ cmake ..
$ make
$ cp myaccel ../../run
```

## 실행
memory bank 역할을 하는 파일 생성 후, 실행
```
$ cd run/memory
$ sh makebanks.sh
$ cd run
$ ./myaccel
```

## 기타 유틸

### device/readfile.c
file의 저장된 값을 확인하기 위한 프로그램
읽고싶은 방식에 따라 적절히 코드를 수정하여 사용
```
$ gcc -o readfile readfile.c
$ ./readfile [file name]
```

### shared_memory_dump.cpp
shared_memory는 레지스터 역할과 명령어, 뱅크정보를 가지고 있음
이것을 dumping하는 프로그램
필요한 내용이 있을 경우 코드에 명령어를 추가하여 기능을 추가
```
$ g++ -o dump shared_memory_dump.cpp
$ ./dump
```
|명령어|역할|
|---|---|
|q|quit : 프로그램을 종료|
|r|register : 레지스터에 저장된 내용들을 확인|
|g|명령어 queue의 상태 확인|
|b|bank : 뱅크 상태 확인|
|s|측정 시간 초기화|
|t|기록된 시간 확인|
