FIND_PACKAGE(opencv)

ADD_LIBRARY(input_image ${CMAKE_CURRENT_LIST_DIR}/input.cpp)
TARGET_LINK_LIBRARIES(input_image opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(paf
            ${CMAKE_CURRENT_LIST_DIR}/paf.cpp
            ${CMAKE_CURRENT_LIST_DIR}/post-process.cpp)
TARGET_LINK_LIBRARIES(paf opencv_core opencv_imgproc opencv_highgui)

ADD_LIBRARY(vis ${CMAKE_CURRENT_LIST_DIR}/vis.cpp)
TARGET_LINK_LIBRARIES(vis opencv_core opencv_imgproc opencv_highgui)

ADD_EXECUTABLE(test_paf
               ${CMAKE_CURRENT_LIST_DIR}/test_paf.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp)
TARGET_LINK_LIBRARIES(test_paf paf vis)

ADD_EXECUTABLE(fake-runner
               ${CMAKE_CURRENT_LIST_DIR}/fake_uff-runner.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp
               ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
TARGET_LINK_LIBRARIES(fake-runner input_image paf vis gflags)


FIND_PACKAGE(gflags)
ADD_EXECUTABLE(process-paf
               ${CMAKE_CURRENT_LIST_DIR}/process-paf.cpp
               ${CMAKE_CURRENT_LIST_DIR}/tracer.cpp)
TARGET_LINK_LIBRARIES(process-paf paf vis gflags)
