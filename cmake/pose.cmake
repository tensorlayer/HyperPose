# Library Name
set(POSE_LIB_NAME pose)

# Compiler Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

# Dependencies(OpenCV & CUDA)
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

EXECUTE_PROCESS(
        COMMAND arch
        COMMAND tr -d '\n'
        OUTPUT_VARIABLE ARCH)
SET(CUDA_RT /usr/local/cuda/targets/${ARCH}-linux)

LINK_DIRECTORIES(${CUDA_RT}/lib)

ADD_LIBRARY(
        ${POSE_LIB_NAME}
        src/*.cpp)
TARGET_LINK_LIBRARIES(
        ${POSE_LIB_NAME}
        cudnn
        cudart
        nvinfer
        nvparsers
        ${OpenCV_LIBS})
TARGET_INCLUDE_DIRECTORIES(
        ${POSE_LIB_NAME} PRIVATE
        ${CUDA_RT}/include
        ${CUDA_RT}/include/crt)
ADD_GLOBAL_DEPS(${POSE_LIB_NAME})