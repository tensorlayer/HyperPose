# Library Name
set(POSE_LIB_NAME hyperpose)

# Compiler Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

# Dependencies(OpenCV & CUDA)
INCLUDE(cmake/cuda.cmake)
FIND_PACKAGE(OpenCV)

ADD_LIBRARY(
        ${POSE_LIB_NAME}
        src/logging.cpp
        src/tensorrt.cpp
        src/paf.cpp
        src/data.cpp
        src/stream.cpp
        src/thread_pool.cpp
        src/human.cpp)
TARGET_LINK_LIBRARIES(
        ${POSE_LIB_NAME}
        cudnn
        cudart
        nvinfer
        nvparsers
        nvonnxparser
        ${OpenCV_LIBS})
TARGET_INCLUDE_DIRECTORIES(
        ${POSE_LIB_NAME} PRIVATE
        ${CUDA_RT}/include
        ${CUDA_RT}/include/crt)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/configuration.h.in ${CMAKE_BINARY_DIR}/configuration.h)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_GLOBAL_DEPS(${POSE_LIB_NAME})
