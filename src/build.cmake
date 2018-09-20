FIND_PACKAGE(opencv)
FIND_PACKAGE(gflags)


ADD_LIBRARY(input_image ${CMAKE_CURRENT_LIST_DIR}/input.cpp)
TARGET_LINK_LIBRARIES(input_image opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(paf
            ${CMAKE_CURRENT_LIST_DIR}/paf.cpp
            ${CMAKE_CURRENT_LIST_DIR}/post-process.cpp)
TARGET_LINK_LIBRARIES(paf opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(vis ${CMAKE_CURRENT_LIST_DIR}/vis.cpp)
TARGET_LINK_LIBRARIES(vis opencv_core opencv_imgproc opencv_highgui)

ADD_EXECUTABLE(fake-runner
               ${CMAKE_CURRENT_LIST_DIR}/fake_uff-runner.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp
               ${CMAKE_CURRENT_LIST_DIR}/uff-runner_main.cpp)
TARGET_LINK_LIBRARIES(fake-runner input_image paf vis gflags)
