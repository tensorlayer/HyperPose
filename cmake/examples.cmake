FIND_PACKAGE(opencv)
FIND_PACKAGE(gflags)
FIND_PACKAGE(Threads REQUIRED)

ADD_LIBRARY(helpers examples/input.cpp examples/vis.cpp)
TARGET_LINK_LIBRARIES(helpers
                      stdtensor
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs)

ADD_LIBRARY(pose-detetor examples/pose_detector.cpp examples/input.cpp)
TARGET_LINK_LIBRARIES(pose-detetor openpose-plus helpers)

ADD_LIBRARY(stream-detetor examples/stream_detector.cpp)
TARGET_LINK_LIBRARIES(stream-detetor openpose-plus helpers)

ADD_EXECUTABLE(example-batch-detector examples/example_batch_detector.cpp)
TARGET_LINK_LIBRARIES(example-batch-detector tracer pose-detetor gflags helpers)

ADD_EXECUTABLE(example-stream-detector examples/example_stream_detector.cpp)
TARGET_LINK_LIBRARIES(
    example-stream-detector tracer stream-detetor gflags Threads::Threads)

ADD_EXECUTABLE(example-live-camera examples/example_live_camera.cpp)
TARGET_LINK_LIBRARIES(example-live-camera
                      tracer
                      stream-detetor
                      gflags
                      opencv_videoio
                      Threads::Threads)
