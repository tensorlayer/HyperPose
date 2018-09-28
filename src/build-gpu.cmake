FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

EXECUTE_PROCESS(COMMAND arch COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH)
SET(CUDA_RT /usr/local/cuda-9.0/targets/${ARCH}-linux)

# FIXME: use TARGET_LINK_DIRECTORIES and TARGET_INCLUDE_DIREC TORIES
LINK_DIRECTORIES(${CUDA_RT}/lib)
INCLUDE_DIRECTORIES(${CUDA_RT}/include ${CUDA_RT}/include/crt)

ADD_LIBRARY(cudnn++ ${CMAKE_CURRENT_LIST_DIR}/cudnn.cpp)
TARGET_LINK_LIBRARIES(cudnn++ cudnn cudart)

ADD_LIBRARY(post-process_cpu ${CMAKE_CURRENT_LIST_DIR}/post-process.cpp)
TARGET_LINK_LIBRARIES(post-process_cpu)

ADD_LIBRARY(paf ${CMAKE_CURRENT_LIST_DIR}/paf.cpp)
TARGET_LINK_LIBRARIES(paf tracer post-process_cpu cudnn++)

ADD_LIBRARY(uff-runner ${CMAKE_CURRENT_LIST_DIR}/uff-runner.cpp)
TARGET_LINK_LIBRARIES(uff-runner nvinfer nvparsers)

ADD_LIBRARY(pose-detetor ${CMAKE_CURRENT_LIST_DIR}/pose_detector.cpp)
TARGET_LINK_LIBRARIES(pose-detetor uff-runner input_image paf)

ADD_LIBRARY(stream-detetor ${CMAKE_CURRENT_LIST_DIR}/stream_detector.cpp)
TARGET_LINK_LIBRARIES(stream-detetor uff-runner input_image paf)

ADD_EXECUTABLE(example ${CMAKE_CURRENT_LIST_DIR}/example.cpp)
TARGET_LINK_LIBRARIES(example tracer pose-detetor vis gflags)

FIND_PACKAGE(Threads REQUIRED)

ADD_EXECUTABLE(example-stream-detector
               ${CMAKE_CURRENT_LIST_DIR}/example_stream_detector.cpp)
TARGET_LINK_LIBRARIES(example-stream-detector
                      tracer
                      stream-detetor
                      vis
                      gflags
                      Threads::Threads)
