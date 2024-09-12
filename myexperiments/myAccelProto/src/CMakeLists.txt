cmake_minimum_required(VERSION 3.10)

project(MyAccelerator CXX)

set(SOURCES
    controller.cpp
    core.cpp
    main.cpp
    utils.cpp
    operations/conv.cpp
    operations/sigmoid.cpp
    operations/mul.cpp
    operations/add.cpp
	operations/maxpool.cpp
	operations/softmax.cpp
	operations/sub.cpp
	operations/div.cpp
)

add_executable(myaccel ${SOURCES})

