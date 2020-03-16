INCLUDE(cmake/cuda.cmake)
FIND_PACKAGE(OpenCV)

ADD_DEFINITIONS(-Ofast -march=native)

ADD_LIBRARY(openpose-plus src/cudnn.cpp src/paf.cpp src/uff_runner.cpp)
TARGET_LINK_LIBRARIES(
    openpose-plus
    cudnn
    cudart
    nvinfer
    nvparsers
    ${OpenCV_LIBS})
TARGET_INCLUDE_DIRECTORIES(openpose-plus PRIVATE ${CUDA_RT}/include
                                                 ${CUDA_RT}/include/crt)
ADD_GLOBAL_DEPS(openpose-plus)
