FIND_PACKAGE(Threads REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

ADD_EXECUTABLE(example-batch-detector examples/example_batch_detector.cpp)
TARGET_LINK_LIBRARIES(example-batch-detector tracer pose-detetor vis gflags)

ADD_EXECUTABLE(example-stream-detector examples/example_stream_detector.cpp)
TARGET_LINK_LIBRARIES(example-stream-detector
                      tracer
                      stream-detetor
                      vis
                      gflags
                      Threads::Threads)

ADD_EXECUTABLE(example-live-camera examples/example_live_camera.cpp)
TARGET_LINK_LIBRARIES(example-live-camera
                      tracer
                      stream-detetor
                      vis
                      gflags
                      opencv_videoio
                      Threads::Threads)
