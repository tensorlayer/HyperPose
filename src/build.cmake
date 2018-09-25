FIND_PACKAGE(opencv)
FIND_PACKAGE(gflags)

ADD_LIBRARY(tracer ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp)

ADD_LIBRARY(input_image ${CMAKE_CURRENT_LIST_DIR}/input.cpp)
TARGET_LINK_LIBRARIES(input_image opencv_core opencv_imgproc opencv_highgui)
TARGET_LINK_LIBRARIES(input_image opencv_imgcodecs) # required on ubuntu 18

ADD_LIBRARY(paf ${CMAKE_CURRENT_LIST_DIR}/paf.cpp
            ${CMAKE_CURRENT_LIST_DIR}/post-process.cpp)
TARGET_LINK_LIBRARIES(paf tracer)

ADD_LIBRARY(vis ${CMAKE_CURRENT_LIST_DIR}/vis.cpp)
TARGET_LINK_LIBRARIES(vis opencv_core opencv_imgproc opencv_highgui)
