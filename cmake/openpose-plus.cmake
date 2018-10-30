FIND_PACKAGE(OpenCV)

ADD_DEFINITIONS(-ffast-math)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
EXECUTE_PROCESS(COMMAND arch COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH)
SET(CUDA_RT /usr/local/cuda-9.0/targets/${ARCH}-linux)

LINK_DIRECTORIES(${CUDA_RT}/lib)

ADD_LIBRARY(openpose-plus src/cudnn.cpp src/paf.cpp src/uff-runner.cpp)
TARGET_LINK_LIBRARIES(openpose-plus
                      cudnn
                      cudart
                      nvinfer
                      nvparsers
                      opencv_core
                      opencv_imgproc)
TARGET_INCLUDE_DIRECTORIES(openpose-plus
                           PRIVATE ${CUDA_RT}/include ${CUDA_RT}/include/crt)
ADD_GLOBAL_DEPS(openpose-plus)
