FIND_PACKAGE(OpenCV)
FIND_PACKAGE(gflags)
FIND_PACKAGE(Threads REQUIRED)

ADD_LIBRARY(helpers examples/input.cpp examples/vis.cpp)
TARGET_LINK_LIBRARIES(helpers
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs)
ADD_GLOBAL_DEPS(helpers)

ADD_LIBRARY(pose-detetor examples/pose_detector.cpp examples/input.cpp)
TARGET_LINK_LIBRARIES(pose-detetor openpose-plus helpers)
ADD_GLOBAL_DEPS(pose-detetor)

ADD_LIBRARY(stream-detetor examples/stream_detector.cpp)
TARGET_LINK_LIBRARIES(stream-detetor openpose-plus helpers)
ADD_GLOBAL_DEPS(stream-detetor)

ADD_EXECUTABLE(example-batch-detector examples/example_batch_detector.cpp)
TARGET_LINK_LIBRARIES(example-batch-detector pose-detetor gflags helpers)
ADD_GLOBAL_DEPS(example-batch-detector)

ADD_EXECUTABLE(example-stream-detector examples/example_stream_detector.cpp)
TARGET_LINK_LIBRARIES(example-stream-detector
                      stream-detetor
                      gflags
                      Threads::Threads)
ADD_GLOBAL_DEPS(example-stream-detector)

ADD_EXECUTABLE(example-live-camera examples/example_live_camera.cpp)
TARGET_LINK_LIBRARIES(example-live-camera
                      stream-detetor
                      gflags
                      opencv_videoio
                      Threads::Threads)
ADD_GLOBAL_DEPS(example-live-camera)
