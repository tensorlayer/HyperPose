# Library Name
set(POSE_LIB_NAME hyperpose)

# Dependencies(OpenCV & CUDA)
INCLUDE(cmake/cuda.cmake)
FIND_PACKAGE(OpenCV REQUIRED)

ADD_LIBRARY(
        ${POSE_LIB_NAME} # SHARED
        src/logging.cpp
        src/tensorrt.cpp
        src/paf.cpp
        src/data.cpp
        src/stream.cpp
        src/thread_pool.cpp
        src/pose_proposal.cpp
        src/human.cpp)

TARGET_LINK_LIBRARIES(
        ${POSE_LIB_NAME}
        cudnn
        cudart
        nvinfer
        nvparsers
        nvonnxparser
        ${OpenCV_LIBS})

TARGET_INCLUDE_DIRECTORIES(${POSE_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDA_RT}/include
        ${CUDA_RT}/include/crt)

SET_TARGET_PROPERTIES(${POSE_LIB_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION})

INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_GLOBAL_DEPS(${POSE_LIB_NAME})
