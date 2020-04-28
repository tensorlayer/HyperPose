INCLUDE(${CMAKE_SOURCE_DIR}/cmake/helpers.cmake)

# [LIBRARY] Pose Detector Lib.
ADD_LIBRARY(pose-detetor
        examples/pose_detector.cpp
        examples/input.cpp)
TARGET_LINK_LIBRARIES(pose-detetor
        openpose-plus
        helpers)
ADD_GLOBAL_DEPS(pose-detetor)

# [LIBRARY] Stream Detector.
ADD_LIBRARY(stream-detetor
        examples/stream_detector.cpp)
TARGET_LINK_LIBRARIES(stream-detetor
        openpose-plus
        helpers)
ADD_GLOBAL_DEPS(stream-detetor)

# [EXAMPLE] Batch Detector.
ADD_EXECUTABLE(example-batch-detector examples/example_batch_detector.cpp)
TARGET_LINK_LIBRARIES(example-batch-detector pose-detetor gflags helpers)
ADD_GLOBAL_DEPS(example-batch-detector)

# [EXAMPLE] Stream Detector.
ADD_EXECUTABLE(example-stream-detector examples/example_stream_detector.cpp)
TARGET_LINK_LIBRARIES(example-stream-detector
        stream-detetor
        gflags
        Threads::Threads)
ADD_GLOBAL_DEPS(example-stream-detector)

# [EXAMPLE] Live Camera.
ADD_EXECUTABLE(example-live-camera examples/example_live_camera.cpp)
TARGET_LINK_LIBRARIES(example-live-camera
        stream-detetor
        gflags
        opencv_videoio
        Threads::Threads)
ADD_GLOBAL_DEPS(example-live-camera)

# [EXAMPLE] Operator API.
ADD_EXECUTABLE(example_operator_api_batched_images examples/example_operator_api_batched_images.cpp)
TARGET_LINK_LIBRARIES(example_operator_api_batched_images helpers swiftpose gflags)
ADD_GLOBAL_DEPS(example_operator_api_batched_images)

# [EXAMPLE] Stream API.
ADD_EXECUTABLE(example_stream_api examples/example_stream_api.cpp)
TARGET_LINK_LIBRARIES(example_stream_api helpers swiftpose gflags)
ADD_GLOBAL_DEPS(example_stream_api)