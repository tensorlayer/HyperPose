FIND_PACKAGE(opencv)
FIND_PACKAGE(gflags)

ADD_LIBRARY(input_image src/input.cpp)
TARGET_LINK_LIBRARIES(input_image
                      tracer
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs)

ADD_LIBRARY(vis src/vis.cpp)
TARGET_LINK_LIBRARIES(vis opencv_core opencv_imgproc opencv_highgui)

ADD_DEFINITIONS(-ffast-math)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

EXECUTE_PROCESS(COMMAND arch COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH)
SET(CUDA_RT /usr/local/cuda-9.0/targets/${ARCH}-linux)

# FIXME: use TARGET_LINK_DIRECTORIES and TARGET_INCLUDE_DIREC TORIES
LINK_DIRECTORIES(${CUDA_RT}/lib)
INCLUDE_DIRECTORIES(${CUDA_RT}/include ${CUDA_RT}/include/crt)

ADD_LIBRARY(cudnn++ src/cudnn.cpp)
TARGET_LINK_LIBRARIES(cudnn++ tracer cudnn cudart)

ADD_LIBRARY(paf src/paf.cpp)
TARGET_LINK_LIBRARIES(paf tracer cudnn++)

ADD_LIBRARY(uff-runner src/uff-runner.cpp)
TARGET_LINK_LIBRARIES(uff-runner tracer nvinfer nvparsers)

ADD_LIBRARY(pose-detetor src/pose_detector.cpp)
TARGET_LINK_LIBRARIES(pose-detetor uff-runner input_image paf)

ADD_LIBRARY(stream-detetor src/stream_detector.cpp)
TARGET_LINK_LIBRARIES(stream-detetor uff-runner input_image paf)

ADD_EXECUTABLE(profile-post-process src/profile_post_process.cpp)
TARGET_LINK_LIBRARIES(profile-post-process
                      tracer
                      gflags
                      paf
                      opencv_core
                      opencv_imgproc)
