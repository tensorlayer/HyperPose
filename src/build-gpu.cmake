# FIXME: use TARGET_LINK_DIRECTORIES and TARGET_INCLUDE_DIRECTORIES
SET(CUDA_RT /usr/local/cuda-9.0/targets/x86_64-linux)

LINK_DIRECTORIES(${CUDA_RT}/lib)
INCLUDE_DIRECTORIES(${CUDA_RT}/include ${CUDA_RT}/include/crt)

ADD_LIBRARY(cuda-buffer ${CMAKE_CURRENT_LIST_DIR}/cuda_buffer.cpp)
TARGET_LINK_LIBRARIES(cuda-buffer cudart)

ADD_LIBRARY(uff-runner ${CMAKE_CURRENT_LIST_DIR}/uff-runner.cpp)
TARGET_LINK_LIBRARIES(uff-runner cuda-buffer nvinfer nvparsers)

ADD_LIBRARY(pose-detetor ${CMAKE_CURRENT_LIST_DIR}/pose_detector.cpp)
TARGET_LINK_LIBRARIES(pose-detetor uff-runner input_image paf)

ADD_EXECUTABLE(uff-runner_main ${CMAKE_CURRENT_LIST_DIR}/uff-runner_main.cpp)
TARGET_LINK_LIBRARIES(uff-runner_main tracer pose-detetor vis gflags)
