FIND_PACKAGE(CUDA REQUIRED)

SET(CUDA_RT /usr/local/cuda-9.0/targets/x86_64-linux)
# FIXME: use TARGET_LINK_DIRECTORIES and TARGET_INCLUDE_DIRECTORIES
LINK_DIRECTORIES(${CUDA_RT}/lib)
INCLUDE_DIRECTORIES(${CUDA_RT}/include ${CUDA_RT}/include/crt)

ADD_LIBRARY(cuda-buffer ${CMAKE_CURRENT_LIST_DIR}/cuda_buffer.cpp)
TARGET_LINK_LIBRARIES(cuda-buffer cudart)

ADD_LIBRARY(cudnn++ ${CMAKE_CURRENT_LIST_DIR}/cudnn.cpp)
TARGET_LINK_LIBRARIES(cudnn++ cudnn)

ADD_LIBRARY(post-process_cpu ${CMAKE_CURRENT_LIST_DIR}/post-process.cpp)
TARGET_LINK_LIBRARIES(post-process_cpu)

ADD_LIBRARY(post-process_gpu ${CMAKE_CURRENT_LIST_DIR}/post-process_gpu.cpp)
TARGET_LINK_LIBRARIES(post-process_gpu post-process_cpu cudnn++)

ADD_LIBRARY(paf ${CMAKE_CURRENT_LIST_DIR}/paf.cpp)
TARGET_LINK_LIBRARIES(paf tracer post-process_cpu post-process_gpu)

ADD_LIBRARY(uff-runner ${CMAKE_CURRENT_LIST_DIR}/uff-runner.cpp)
TARGET_LINK_LIBRARIES(uff-runner cuda-buffer nvinfer nvparsers)

ADD_LIBRARY(pose-detetor ${CMAKE_CURRENT_LIST_DIR}/pose_detector.cpp)
TARGET_LINK_LIBRARIES(pose-detetor uff-runner input_image paf)

ADD_EXECUTABLE(uff-runner_main ${CMAKE_CURRENT_LIST_DIR}/uff-runner_main.cpp)
TARGET_LINK_LIBRARIES(uff-runner_main tracer pose-detetor vis gflags)
