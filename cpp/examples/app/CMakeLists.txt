CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
PROJECT(see-pose)

FIND_PACKAGE(opencv)

SET(CMAKE_CXX_STANDARD 17)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

SET(TF_ROOT ${CMAKE_SOURCE_DIR}/../../tensorflow)

INCLUDE_DIRECTORIES(${TF_ROOT})
LINK_DIRECTORIES(${TF_ROOT}/bazel-bin/tensorflow/examples/pose-inference)

FILE(GLOB srcs src/*.cc)
ADD_EXECUTABLE(see-pose ${srcs})

TARGET_LINK_LIBRARIES(see-pose
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs
                      pose-detector)
