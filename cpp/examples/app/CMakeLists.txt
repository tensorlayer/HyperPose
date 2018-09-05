CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
PROJECT(see-pose)

SET(PROJECT_ROOT ${CMAKE_SOURCE_DIR}/../../..)
include(cmake/pafprocess.cmake)
include(cmake/tensorflow.cmake)
include(cmake/3rdparty.cmake)

# FIND_PACKAGE(opencv REQUIRED)  # FIXME: adding REQUIRED cause error on Linux
FIND_PACKAGE(opencv)
FIND_PACKAGE(Threads)

SET(CMAKE_CXX_STANDARD 17)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

FILE(GLOB srcs src/*.cc)
ADD_EXECUTABLE(see-pose ${srcs})

TARGET_LINK_LIBRARIES(see-pose
                      gflags
                      glog
                      Threads::Threads
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs
                      pafprocess
                      pose-detector)
