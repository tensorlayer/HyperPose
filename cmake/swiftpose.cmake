# TODO: Replace openpose-plus with this when code refactoring is done.

# Library Name
set(POSE_LIB_NAME swiftpose)

# Compiler Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

# Dependencies(OpenCV & CUDA)
INCLUDE(cmake/cuda.cmake)
FIND_PACKAGE(OpenCV)

ADD_LIBRARY(
        ${POSE_LIB_NAME}
        src/paf_.cpp
        src/logging.cpp
        src/tensorrt.cpp
        src/paf_.cpp
        src/data.cpp
        src/stream.cpp
        src/viz.cpp)
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
