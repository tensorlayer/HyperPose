# ADD_EXECUTABLE(hello ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

FIND_PACKAGE(opencv)

ADD_LIBRARY(input_image ${CMAKE_CURRENT_LIST_DIR}/input.cpp)
TARGET_LINK_LIBRARIES(input_image opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(paf ${CMAKE_CURRENT_LIST_DIR}/pafprocess/pafprocess.cpp
            ${CMAKE_CURRENT_LIST_DIR}/paf.cpp)
TARGET_LINK_LIBRARIES(paf opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(vis ${CMAKE_CURRENT_LIST_DIR}/vis.cpp)
TARGET_LINK_LIBRARIES(vis opencv_core opencv_imgproc opencv_highgui)

ADD_EXECUTABLE(runner
               ${CMAKE_CURRENT_LIST_DIR}/uff-runner.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp
               ${CMAKE_CURRENT_LIST_DIR}/main.cpp
               ${CMAKE_CURRENT_LIST_DIR}/cuda_buffer.cpp)
TARGET_LINK_LIBRARIES(
    runner input_image paf vis nvinfer cudart nvparsers)
