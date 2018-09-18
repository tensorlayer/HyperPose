# FIXME: use TARGET_LINK_DIRECTORIES and TARGET_INCLUDE_DIRECTORIES
LINK_DIRECTORIES(/usr/local/cuda-9.0/targets/x86_64-linux/lib)
INCLUDE_DIRECTORIES(/usr/local/cuda-9.0/targets/x86_64-linux/include
                    /usr/local/cuda-9.0/targets/x86_64-linux/include/crt)

ADD_EXECUTABLE(runner
               ${CMAKE_CURRENT_LIST_DIR}/uff-runner.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp
               ${CMAKE_CURRENT_LIST_DIR}/main.cpp
               ${CMAKE_CURRENT_LIST_DIR}/cuda_buffer.cpp)
TARGET_LINK_LIBRARIES(runner input_image paf vis gflags nvinfer cudart nvparsers)
