load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

tf_cc_shared_object(
    name = "libtf-runner.so",
    srcs = [
        "tf-runner.cpp",
        "tf-runner.h",
    ],
    linkopts = ["-shared"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)
