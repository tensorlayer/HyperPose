load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "pose-inference",
    srcs = ["main.cc"],
    deps = [
        ":pose-detector",
    ],
)

cc_library(
    name = "pose-detector",
    srcs = [
        "pose-detector.cc",
        "pose-detector.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)
