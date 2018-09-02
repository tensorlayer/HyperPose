SET(TF_ROOT ${CMAKE_SOURCE_DIR}/../../tensorflow)
INCLUDE_DIRECTORIES(${TF_ROOT})
LINK_DIRECTORIES(${TF_ROOT}/bazel-bin/tensorflow/examples/pose-inference)
