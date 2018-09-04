load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_shared_object")

tf_cc_shared_object(
    name = "libpose-detector.so",
    srcs = [
        "pose-detector.cc",
        "pose-detector.h",
    ],
    linkopts = ["-shared"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)
